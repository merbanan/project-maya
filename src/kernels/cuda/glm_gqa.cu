// src/kernels/cuda/glm_gqa.cu - hy_v3's grouped-query attention (glm_fast.hpp: GqaKv, gqa_prep, gqa_attn,
// gqa_attn_rows).  The arithmetic is llama.cpp's hy-v3 graph (src/models/hy-v3.cpp): per-head RMS norms on q and k
// (llm_build_norm over head_dim), NeoX rope over the whole head (ggml-cuda/rope.cu's rope_neox: theta = pos *
// base^(-2i/n_dims), the pair (i, i + n_dims/2)), K and V in a cache, softmax(q.k / sqrt(head_dim)) . v.
//
// The cache (GqaKv): POSITION-MAJOR records - a position's record holds every layer's K rows then V rows (n_kv heads
// each); a row is FP16 (256 bytes), INT8 codes + one FP16 scale per 32 values (symmetric, absmax / 127: 136 bytes), or
// - K8V4: INT8 K and 4-bit V - 4-bit codes, two a byte, + a scale per 32 (absmax / 7: 72 bytes).  Positions below p0
// are in region a, the rest in region b (the expert pool's lendable tail, lent to the cache as the context grows; its
// records run DOWN from b, so the prompt path's buffers at the tail's start meet them only near a full context).  K8V4's V
// rows are rotated by the orthonormal 128-point Walsh-Hadamard transform H before they are quantized (outliers spread
// over the row, as Strata's K8V4); attention is linear in V, so its output is rotated back by H (H H = I) per head.
//
// The TurboQuant modes (TQ4: 4-bit K and V; TQ6V4: 6-bit K, 4-bit V; TQ4V3: 4-bit K, 3-bit V; TQ6V3: 6-bit K, 3-bit
// V; no QJL residual): a
// row is rotated by R = H D (D a fixed random sign per dimension), each rotated value - scaled to unit variance by the
// row's norm - takes the nearest level of the Lloyd-Max codebook of a unit Gaussian (what a rotated coordinate is close
// to), and the row keeps one F32 scale: the least-squares fit of its values to the levels.  K's rotation is undone on q
// instead (q.k = Rq.Rk: q is rotated by R in gqa_prep), V's on the attention's output (o = D H o').  Codes: 4-bit two
// a byte; 3-bit eight in 3 bytes; 6-bit the low 4 bits as 4-bit's then the high 2 bits four a byte (aligned loads); the
// scale after them (52 / 68 / 100 bytes a row).
#include "strata/kernels/glm_fast.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#if !defined(STRATA_USE_HIP)
#include <mma.h>
#endif

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::kernels::glmf {
namespace {

constexpr int kD = 128;           // head_dim (hy3_geometry requires it)
constexpr int kChunk = 256;       // positions per split of the decode attention
constexpr int kMaxGroup = 16;     // query heads per K/V head

// a row's format: FP16, INT8 (a scale per 32), K8V4's 4-bit V (a scale per 32), TurboQuant 3 / 4 / 6 bits
enum : int { kF16 = 0, kI8 = 1, kL4 = 2, kT3 = 3, kT4 = 4, kT6 = 5 };
__host__ __device__ __forceinline__ int row_bytes(int kind) {
    return kind == kF16 ? kD * 2 : kind == kI8 ? kD + (kD / 32) * 2 : kind == kL4 ? kD / 2 + (kD / 32) * 2
         : kind == kT3 ? kD * 3 / 8 + 4 : kind == kT4 ? kD / 2 + 4 : kD * 6 / 8 + 4;
}
__host__ __device__ __forceinline__ int tq_levels(int kind) { return kind == kT3 ? 8 : kind == kT4 ? 16 : 64; }
// the Lloyd-Max levels of a unit Gaussian (MSE 0.0346 / 0.0095 / 0.00064)
__constant__ float c_cb3[8] = {-2.151946f, -1.343909f, -0.756005f, -0.245094f, 0.245094f, 0.756005f, 1.343909f,
                               2.151946f};
__constant__ float c_cb4[16] = {-2.732590f, -2.069017f, -1.618046f, -1.256231f, -0.942340f, -0.656759f, -0.388048f,
                                -0.128395f, 0.128395f, 0.388048f, 0.656759f, 0.942340f, 1.256231f, 1.618046f,
                                2.069017f, 2.732590f};
__constant__ float c_cb6[64] = {
    -3.744101f, -3.240437f, -2.917407f, -2.672274f, -2.471305f, -2.298981f, -2.146810f, -2.009611f, -1.883977f,
    -1.767542f, -1.658589f, -1.555831f, -1.458276f, -1.365141f, -1.275794f, -1.189720f, -1.106488f, -1.025736f,
    -0.947155f, -0.870477f, -0.795468f, -0.721922f, -0.649656f, -0.578503f, -0.508314f, -0.438950f, -0.370283f,
    -0.302193f, -0.234567f, -0.167297f, -0.100278f, -0.033410f, 0.033410f,  0.100278f,  0.167297f,  0.234567f,
    0.302193f,  0.370283f,  0.438950f,  0.508314f,  0.578503f,  0.649656f,  0.721922f,  0.795468f,  0.870477f,
    0.947155f,  1.025736f,  1.106488f,  1.189720f,  1.275794f,  1.365141f,  1.458276f,  1.555831f,  1.658589f,
    1.767542f,  1.883977f,  2.009611f,  2.146810f,  2.298981f,  2.471305f,  2.672274f,  2.917407f,  3.240437f,
    3.744101f};
__device__ __forceinline__ const float* tq_cb(int kind) { return kind == kT3 ? c_cb3 : kind == kT4 ? c_cb4 : c_cb6; }
// D: a fixed random sign per dimension (128 bits)
__constant__ unsigned c_sign[4] = {0x9e3779b9u, 0x7f4a7c15u, 0xf39cc060u, 0x5ced8f29u};
__device__ __forceinline__ float rsign(int d) { return (c_sign[d >> 5] >> (d & 31)) & 1u ? -1.0f : 1.0f; }
// the codebooks of K and V (64 levels at most each) into shared memory: a kernel's lookups then cost a shared read
// (a table of the 32 positive levels and a sign bit - no bank conflicts - measured slower: the sign costs more ALU)
__device__ __forceinline__ void load_cbs(float (*scb)[64], int kk, int vk, int t, int nt) {
    for (int i = t; i < 128; i += nt) {
        const int kind = i < 64 ? kk : vk, j = i & 63;
        scb[i >> 6][j] = kind >= kT3 && j < tq_levels(kind) ? tq_cb(kind)[j] : 0.0f;
    }
}

__device__ __forceinline__ float wsum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float wmax(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// a K or V row (position, head): its record (region a below p0, else b; a streamed view's cached page), the layer's
// part of it - laid out [K codes][V codes][K scales][V scales], so every row's codes are 16-byte aligned (whole-row
// vector loads) - the head's codes at coff + h cb and scales at soff + h sb
__host__ __device__ __forceinline__ int code_bytes(int kind) {
    return kind == kF16 ? kD * 2 : kind == kI8 ? kD : kind == kL4 ? kD / 2 : kind == kT3 ? kD * 3 / 8 : kind == kT4 ? kD / 2
         : kD * 6 / 8;
}
__host__ __device__ __forceinline__ int scale_bytes(int kind) {
    return kind == kF16 ? 0 : kind == kI8 || kind == kL4 ? (kD / 32) * 2 : 4;
}
struct Rows {
    const uint8_t* a;
    const uint8_t* b;
    long long p0, rec;
    int coff, cb, soff, sb, kind;   // kind: kF16 .. kT6
    const float* lv;     // a TurboQuant kind's levels (shared memory)
    const int* ptab;     // KV streaming: the rows of the cached pages (page table -> slots, rec bytes a position)
    const uint8_t* slots;
    __device__ __forceinline__ const uint8_t* base(long long pos) const {
        if (ptab != nullptr) return slots + ((size_t) ptab[pos / kGqaPage] * kGqaPage + (size_t) (pos % kGqaPage)) * rec;
        return pos < p0 ? a + pos * rec : b - (pos - p0) * rec;   // (region b grows down)
    }
    __device__ __forceinline__ const uint8_t* codes(const uint8_t* bp, int h) const { return bp + coff + h * cb; }
    __device__ __forceinline__ const uint8_t* scales(const uint8_t* bp, int h) const { return bp + soff + h * sb; }
    __device__ __forceinline__ void load4(long long pos, int h, int d, float& x0, float& x1, float& x2, float& x3) const {
        load4_b(base(pos), h, d, x0, x1, x2, x3);
    }
    // values d .. d + 3 of head h's row in the record at bp
    __device__ __forceinline__ void load4_b(const uint8_t* bp, int h, int d, float& x0, float& x1, float& x2,
                                            float& x3) const {
        load4_cs(codes(bp, h), scales(bp, h), d, x0, x1, x2, x3);
    }
    // ... of a row staged in shared memory: its codes, then its scales
    __device__ __forceinline__ void load4_s(const uint8_t* srow, int d, float& x0, float& x1, float& x2, float& x3) const {
        load4_cs(srow, srow + cb, d, x0, x1, x2, x3);
    }
    // ... from the row's codes rp and scales sp
    __device__ __forceinline__ void load4_cs(const uint8_t* rp, const uint8_t* sp, int d, float& x0, float& x1, float& x2,
                                             float& x3) const {
        if (kind >= kT3) {
            const float sc = *(const float*) sp;
            unsigned c0, c1, c2, c3;
            if (kind == kT4) {
                const unsigned qq = *(const uint16_t*) (rp + d / 2);
                c0 = qq & 15; c1 = (qq >> 4) & 15; c2 = (qq >> 8) & 15; c3 = qq >> 12;
            } else if (kind == kT3) {   // eight codes in 3 bytes: d's four from bit 12 ((d / 4) & 1)
                const uint8_t* g = rp + 3 * (d >> 3);
                const unsigned u = ((unsigned) g[0] | ((unsigned) g[1] << 8) | ((unsigned) g[2] << 16)) >> (12 * ((d >> 2) & 1));
                c0 = u & 7; c1 = (u >> 3) & 7; c2 = (u >> 6) & 7; c3 = (u >> 9) & 7;
            } else {                    // 6-bit: the low 4 bits as nibbles (64 bytes), then the high 2 bits, 4 a byte
                const unsigned lo = *(const uint16_t*) (rp + d / 2), hi = rp[kD / 2 + d / 4];
                c0 = (lo & 15) | ((hi & 3) << 4);
                c1 = ((lo >> 4) & 15) | (((hi >> 2) & 3) << 4);
                c2 = ((lo >> 8) & 15) | (((hi >> 4) & 3) << 4);
                c3 = (lo >> 12) | ((hi >> 6) << 4);
            }
            x0 = lv[c0] * sc; x1 = lv[c1] * sc; x2 = lv[c2] * sc; x3 = lv[c3] * sc;
        } else if (kind == kL4) {
            const uint16_t qq = *(const uint16_t*) (rp + d / 2);
            const float sc = __half2float(((const __half*) sp)[d / 32]);
            x0 = (float) ((int) (qq & 15) - 8) * sc;
            x1 = (float) ((int) ((qq >> 4) & 15) - 8) * sc;
            x2 = (float) ((int) ((qq >> 8) & 15) - 8) * sc;
            x3 = (float) ((int) ((qq >> 12) & 15) - 8) * sc;
        } else if (kind == kI8) {
            const char4 q = *(const char4*) (rp + d);
            const float sc = __half2float(((const __half*) sp)[d / 32]);
            x0 = q.x * sc; x1 = q.y * sc; x2 = q.z * sc; x3 = q.w * sc;
        } else {
            const __half2* hp = (const __half2*) ((const __half*) rp + d);
            const float2 u = __half22float2(hp[0]), w = __half22float2(hp[1]);
            x0 = u.x; x1 = u.y; x2 = w.x; x3 = w.y;
        }
    }
};
// (cached: the rows of the cached pages - a streamed view's selection; else the view's own records)
__device__ __forceinline__ Rows k_rows(const GqaKv& kv, const float* lv = nullptr, bool cached = false) {
    const int ck = code_bytes(kv.kk), cv = code_bytes(kv.vk);
    return {(const uint8_t*) kv.a, (const uint8_t*) kv.b, kv.p0, kv.rec, kv.k_off, ck,
            kv.k_off + kv.n_kv * (ck + cv), scale_bytes(kv.kk), kv.kk, lv, cached ? kv.ptab : nullptr, kv.slots};
}
__device__ __forceinline__ Rows v_rows(const GqaKv& kv, const float* lv = nullptr, bool cached = false) {
    const int ck = code_bytes(kv.kk), cv = code_bytes(kv.vk);
    return {(const uint8_t*) kv.a, (const uint8_t*) kv.b, kv.p0, kv.rec, kv.k_off + kv.n_kv * ck, cv,
            kv.k_off + kv.n_kv * (ck + cv + scale_bytes(kv.kk)), scale_bytes(kv.vk), kv.vk, lv,
            cached ? kv.ptab : nullptr, kv.slots};
}
// the decode's key index i: position i, or (a streamed view's selection) position j of the (i / kGqaPage)-th selected
// page; n_idx keys in all, those at n_pos or past it (the last page's tail) masked
struct Keys {
    const int* sel;
    int n_idx, n_pos;
    __device__ __forceinline__ long long pos(int i) const {
        return sel != nullptr ? (long long) sel[i / kGqaPage] * kGqaPage + i % kGqaPage : i;
    }
    __device__ __forceinline__ bool ok(int i) const { return i < n_idx && pos(i) < n_pos; }
};
__device__ __forceinline__ Keys keys_of(const GqaKv& kv, int n_pos) {
    return {kv.sel, kv.sel != nullptr ? *kv.nsel * kGqaPage : n_pos, n_pos};
}
// the rotations: K by R = H D (TurboQuant kinds); V by H (K8V4's 4-bit: 1) or R (2)
__host__ __device__ __forceinline__ bool k_rot(const GqaKv& kv) { return kv.kk >= kT3; }
__host__ __device__ __forceinline__ int v_rot(const GqaKv& kv) { return kv.vk == kL4 ? 1 : kv.vk >= kT3 ? 2 : 0; }
// the sum of v over the block's 128 threads (every thread calls it; red: 4 floats of shared memory)
__device__ __forceinline__ float bsum128(float v, float* red, int t) {
    v = wsum(v);
    __syncthreads();
    if ((t & 31) == 0) red[t >> 5] = v;
    __syncthreads();
    return red[0] + red[1] + red[2] + red[3];
}

// the orthonormal 128-point Walsh-Hadamard transform of s (shared, 128 values) by 128 threads, in place
__device__ __forceinline__ void fwht128(float* s, int t) {
#pragma unroll
    for (int h = 1; h < kD; h <<= 1) {
        __syncthreads();
        const float a = s[t], b = s[t ^ h];
        __syncthreads();
        s[t] = (t & h) ? b - a : a + b;
    }
    __syncthreads();
    s[t] *= 0.08838834764831845f;   // 1 / sqrt(128)
    __syncthreads();
}

// one row of 128 values into the cache (kD threads, value t from each): FP16, INT8 with a scale per 32 (a warp), or
// 4-bit codes, two a byte (the even thread writes the pair), with a scale per 32
__device__ __forceinline__ void store_row(const Rows& R, long long pos, int h, int t, float val) {
    const uint8_t* bp = R.base(pos);
    uint8_t* rp = (uint8_t*) R.codes(bp, h);
    uint8_t* sp = (uint8_t*) R.scales(bp, h);
    if (R.kind >= kT3) {   // TurboQuant (a rotated row): the nearest level of val * sqrt(128) / |row|, the LS scale
        __shared__ float red[4];
        const float* cb = tq_cb(R.kind);
        const int nl = tq_levels(R.kind);
        const float ss = bsum128(val * val, red, t);
        const float y = ss > 0.0f ? val * sqrtf((float) kD / ss) : 0.0f;
        unsigned code = 0;
        for (int i = 0; i + 1 < nl; ++i) code += y > 0.5f * (cb[i] + cb[i + 1]);
        const float c = cb[code];
        const float num = bsum128(val * c, red, t), den = bsum128(c * c, red, t);
        if (R.kind == kT4 || R.kind == kT6) {   // (6-bit: the low nibbles as 4-bit's, the high 2 bits 4 a byte after)
            const unsigned lo = code & 15, cn = __shfl_xor_sync(0xffffffffu, lo, 1);
            if ((t & 1) == 0) rp[t / 2] = (uint8_t) (lo | (cn << 4));
            if (R.kind == kT6) {
                unsigned u = (code >> 4) << (2 * (t & 3));
                u |= __shfl_xor_sync(0xffffffffu, u, 1);
                u |= __shfl_xor_sync(0xffffffffu, u, 2);
                if ((t & 3) == 0) rp[kD / 2 + t / 4] = (uint8_t) u;
            }
        } else {   // 3-bit: eight a group of 3 bytes
            const int per = 8, bits = 3;
            unsigned u = code << (bits * (t & (per - 1)));
            for (int o = 1; o < per; o <<= 1) u |= __shfl_xor_sync(0xffffffffu, u, o);
            if ((t & (per - 1)) == 0) {
                uint8_t* g = rp + 3 * (t / per);
                g[0] = (uint8_t) u;
                g[1] = (uint8_t) (u >> 8);
                g[2] = (uint8_t) (u >> 16);
            }
        }
        if (t == 0) *(float*) sp = den > 0.0f ? num / den : 0.0f;
    } else if (R.kind == kL4) {
        const float amax = wmax(fabsf(val));
        const float sc = amax / 7.0f;
        const int q = sc > 0.0f ? max(-7, min(7, __float2int_rn(val / sc))) : 0;
        const int qn = __shfl_xor_sync(0xffffffffu, q, 1);
        if ((t & 1) == 0) rp[t / 2] = (uint8_t) ((q + 8) | ((qn + 8) << 4));
        if ((t & 31) == 0) ((__half*) sp)[t / 32] = __float2half(sc);
    } else if (R.kind == kI8) {
        const float amax = wmax(fabsf(val));
        const float sc = amax / 127.0f;
        const int q = sc > 0.0f ? __float2int_rn(val / sc) : 0;
        rp[t] = (uint8_t) (int8_t) max(-127, min(127, q));
        if ((t & 31) == 0) ((__half*) sp)[t / 32] = __float2half(sc);
    } else {
        ((__half*) rp)[t] = __float2half(val);
    }
}

// grid (T, n_head + 2 n_kv), kD threads: blockIdx.y < n_head a q head (norm + rope in place), then the k heads (norm +
// rope into the K cache at p0 + row), then the v heads (into the V cache).  q / k / v rows qld / kld floats apart.
__global__ void __launch_bounds__(kD) gqa_prep_kernel(float* __restrict__ q, const float* __restrict__ k,
                                                      const float* __restrict__ v, int qld, int kld,
                                                      const float* __restrict__ qn, const float* __restrict__ kn,
                                                      float eps, int p0, float theta_scale, int n_head, GqaKv kv) {
    __shared__ float sy[kD];
    __shared__ float sred[kD / 32];
    const int row = blockIdx.x, b = blockIdx.y, t = threadIdx.x, pos = p0 + row;
    const int n_kv = kv.n_kv;
    if (b >= n_head + n_kv) {                                  // a v head
        const int h = b - n_head - n_kv;
        float val = v[(size_t) row * kld + (size_t) h * kD + t];
        if (const int vr = v_rot(kv)) {   // K8V4: the row rotated by H before its 4-bit codes; TurboQuant: by H D
            sy[t] = vr == 2 ? val * rsign(t) : val;
            fwht128(sy, t);
            val = sy[t];
        }
        store_row(v_rows(kv), pos, h, t, val);
        if (kv.ptab != nullptr && kv.ptab[pos / kGqaPage] >= 0) store_row(v_rows(kv, nullptr, true), pos, h, t, val);
        return;
    }
    const bool is_q = b < n_head;
    const int h = is_q ? b : b - n_head;
    const float x = is_q ? q[(size_t) row * qld + (size_t) h * kD + t] : k[(size_t) row * kld + (size_t) h * kD + t];
    float ss = wsum(x * x);
    if ((t & 31) == 0) sred[t >> 5] = ss;
    __syncthreads();
    ss = 0.0f;
#pragma unroll
    for (int i = 0; i < kD / 32; ++i) ss += sred[i];
    const float y = x * rsqrtf(ss / (float) kD + eps) * (is_q ? qn[t] : kn[t]);
    sy[t] = y;
    __syncthreads();
    // rope_neox: x0 = y[i], x1 = y[i + n_dims/2], theta = pos * theta_scale^i
    const int i = t & (kD / 2 - 1);
    const float theta = (float) pos * powf(theta_scale, (float) i);
    const float c = cosf(theta), s = sinf(theta);
    const float x0 = sy[i], x1 = sy[i + kD / 2];
    float o = t < kD / 2 ? x0 * c - x1 * s : x0 * s + x1 * c;
    if (k_rot(kv)) {   // TurboQuant K: k and q both rotated by H D (q.k unchanged)
        __syncthreads();   // (every thread has read its rope pair)
        sy[t] = o * rsign(t);
        fwht128(sy, t);
        o = sy[t];
    }
    if (is_q) {
        q[(size_t) row * qld + (size_t) h * kD + t] = o;
    } else {
        store_row(k_rows(kv), pos, h, t, o);
        // (a streamed view: the host record always, the cached page's copy as well when it is cached)
        if (kv.ptab != nullptr && kv.ptab[pos / kGqaPage] >= 0) store_row(k_rows(kv, nullptr, true), pos, h, t, o);
    }
}

// ---- decode: split-K over [0, n_pos) for the G query heads of one K/V head; per (head, split) the running max m,
// the sum l and the unnormalised output o[kD].  256 threads = 8 warps.
__global__ void __launch_bounds__(256) gqa_attn_kernel(const float* __restrict__ q, GqaKv kv, int n_pos, int n_head,
                                                       float scale, float* __restrict__ part) {
    __shared__ float sq[kMaxGroup * kD];
    __shared__ float sp[kMaxGroup * kChunk];
    __shared__ float sm[kMaxGroup], sl[kMaxGroup];
    const int n_kv = kv.n_kv, kvh = blockIdx.x, split = blockIdx.y, G = n_head / n_kv;
    const int t = threadIdx.x, lane = t & 31, w = t >> 5;
    const Keys ks = keys_of(kv, n_pos);
    const int p0 = split * kChunk, np = min(kChunk, ks.n_idx - p0);
    __shared__ float scb[2][64];
    load_cbs(scb, kv.kk, kv.vk, t, blockDim.x);
    const bool cached = kv.sel != nullptr;
    const Rows K = k_rows(kv, scb[0], cached), V = v_rows(kv, scb[1], cached);
    for (int i = t; i < G * kD; i += blockDim.x) sq[i] = q[(size_t) (kvh * G) * kD + i] * scale;
    __syncthreads();
    for (int j = w; j < np; j += 8) {
        if (!ks.ok(p0 + j)) {   // (a page's tail past n_pos)
            if (lane == 0)
                for (int g = 0; g < G; ++g) sp[g * kChunk + j] = -INFINITY;
            continue;
        }
        float k0, k1, k2, k3;
        K.load4(ks.pos(p0 + j), kvh, 4 * lane, k0, k1, k2, k3);
        for (int g = 0; g < G; ++g) {
            const float* qg = sq + g * kD + 4 * lane;
            float d = qg[0] * k0 + qg[1] * k1 + qg[2] * k2 + qg[3] * k3;
            d = wsum(d);
            if (lane == 0) sp[g * kChunk + j] = d;
        }
    }
    __syncthreads();
    for (int g = w; g < G; g += 8) {
        float m = -INFINITY;
        for (int j = lane; j < np; j += 32) m = fmaxf(m, sp[g * kChunk + j]);
        m = wmax(m);
        float l = 0.0f;
        for (int j = lane; j < np; j += 32) {
            const float e = sp[g * kChunk + j] == -INFINITY ? 0.0f : __expf(sp[g * kChunk + j] - m);
            sp[g * kChunk + j] = e;
            l += e;
        }
        l = wsum(l);
        if (lane == 0) {
            sm[g] = m;
            sl[g] = l;
        }
    }
    __syncthreads();
    const int d4 = 4 * lane;
    for (int g = w; g < G; g += 8) {
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        for (int j = 0; j < np; ++j) {
            const float pj = sp[g * kChunk + j];
            if (pj == 0.0f) continue;   // (masked keys: their rows may be stale)
            float v0, v1, v2, v3;
            V.load4(ks.pos(p0 + j), kvh, d4, v0, v1, v2, v3);
            a0 += pj * v0;
            a1 += pj * v1;
            a2 += pj * v2;
            a3 += pj * v3;
        }
        const int head = kvh * G + g;
        float* o = part + ((size_t) head * gridDim.y + split) * (kD + 2);
        o[d4] = a0;
        o[d4 + 1] = a1;
        o[d4 + 2] = a2;
        o[d4 + 3] = a3;
        if (lane == 0) {
            o[kD] = sm[g];
            o[kD + 1] = sl[g];
        }
    }
}

struct Q8_1 {   // ggml's block_q8_1: the scale and the sum (FP16), 32 int8
    __half2 ds;
    int8_t qs[32];
};
// head h's output from its n_split parts (merged over the splits' maxima), by kD threads: out (F32) and, when q8 is
// given, its q8_1 (quantize_q8_1's arithmetic: a warp a block of 32).  so: kD floats of shared memory
__device__ __forceinline__ void combine_head(const float* __restrict__ part, int n_split, int h, int t, int vrot,
                                             float* so, float* __restrict__ out, Q8_1* __restrict__ q8) {
    const float* ph = part + (size_t) h * n_split * (kD + 2);
    float M = -INFINITY;
    for (int s = 0; s < n_split; ++s) M = fmaxf(M, ph[(size_t) s * (kD + 2) + kD]);
    float acc = 0.0f, den = 0.0f;
    for (int s = 0; s < n_split; ++s) {
        const float* ps = ph + (size_t) s * (kD + 2);
        const float f = __expf(ps[kD] - M);
        acc += f * ps[t];
        den += f * ps[kD + 1];
    }
    float v = acc / den;
    if (vrot) {   // the output was made of rotated V rows - rotate it back (H, or D H)
        so[t] = v;
        fwht128(so, t);
        v = vrot == 2 ? so[t] * rsign(t) : so[t];
        __syncthreads();   // (so: the next head's)
    }
    out[(size_t) h * kD + t] = v;
    if (q8 != nullptr) {
        const int lane = t & 31;
        const float amax = wmax(fabsf(v)), sum = wsum(v);
        const float d = fminf(amax / 127.0f, 65504.0f);
        Q8_1* dst = q8 + (size_t) h * (kD / 32) + t / 32;
        dst->qs[lane] = amax == 0.0f ? 0 : (int8_t) fmaxf(-127.0f, fminf(127.0f, roundf(v / d)));
        if (lane == 0) dst->ds = make_half2(d, fmaxf(-65504.0f, fminf(65504.0f, sum)));
    }
}

// one head a block: its output merged from the splits' parts, F32 and its q8_1 (the out-projection's input)
__global__ void __launch_bounds__(kD) gqa_combine_kernel(const float* __restrict__ part, int n_split,
                                                         float* __restrict__ out, Q8_1* __restrict__ q8, int vrot) {
    __shared__ float so[kD];
    combine_head(part, n_split, blockIdx.x, threadIdx.x, vrot, so, out, q8);
}

// K8V4, the prompt path: each FP16 output row (T rows of n_head heads, old halves apart) rotated back by H
__global__ void __launch_bounds__(kD) gqa_unrotate_rows_kernel(__half* __restrict__ out, int n_head, int old, int vrot) {
    __shared__ float so[kD];
    const int row = blockIdx.x, h = blockIdx.y, t = threadIdx.x;
    __half* o = out + (size_t) row * old + (size_t) h * kD;
    so[t] = __half2float(o[t]);
    fwht128(so, t);
    o[t] = __float2half(vrot == 2 ? so[t] * rsign(t) : so[t]);
}

// ---- a prompt chunk: causal attention of rows [0, T) (positions p0 + row) over the cache [0, p0 + row].  Block (32
// query rows, one query head), 256 threads = 8 warps; warp w owns rows w, w + 8, w + 16, w + 24 (scores: a lane per
// key of a 32-key tile; output: 4 dims a lane) with their running max / sum in registers (flash attention's online
// softmax).  K / V tiles go through shared memory as FP16 (a K row padded to 65 half2: the lanes read different rows of
// the same column without bank conflicts).
constexpr int kQT = 32, kKT = 32;
__global__ void __launch_bounds__(256) gqa_attn_rows_kernel(const float* __restrict__ q, int qld, GqaKv kv, int p0,
                                                            int T, int n_head, float scale, __half* __restrict__ out,
                                                            int old) {
    __shared__ float sQ[kQT][kD];
    __shared__ __half2 sK[kKT][kD / 2 + 1];
    __shared__ __half2 sV[kKT][kD / 2];
    __shared__ float sP[kQT][kKT + 1];
    const int h = blockIdx.y, r0 = blockIdx.x * kQT, t = threadIdx.x, lane = t & 31, w = t >> 5;
    const int n_kv = kv.n_kv, kvh = h / (n_head / n_kv);
    __shared__ float scb[2][64];
    load_cbs(scb, kv.kk, kv.vk, t, 256);
    const Rows K = k_rows(kv, scb[0]), V = v_rows(kv, scb[1]);
    for (int i = t; i < kQT * kD; i += 256) {
        const int r = i / kD, d = i % kD;
        sQ[r][d] = r0 + r < T ? q[(size_t) (r0 + r) * qld + (size_t) h * kD + d] * scale : 0.0f;
    }
    float m[4], l[4], acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        m[i] = -INFINITY;
        l[i] = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;
    }
    const int last = p0 + min(r0 + kQT, T) - 1;   // the last position any row of this tile sees
    for (int k0 = 0; k0 <= last; k0 += kKT) {
        __syncthreads();   // (the previous tile's sK / sV / sP reads are done)
        // the tile's K and V rows: 32 rows x 32 groups of 4 values, one group a thread a pass
        for (int i = t; i < kKT * (kD / 4); i += 256) {
            const int r = i / (kD / 4), d = (i % (kD / 4)) * 4;
            float a = 0.f, b = 0.f, c = 0.f, e = 0.f, va = 0.f, vb = 0.f, vc = 0.f, ve = 0.f;
            if (k0 + r <= last) {
                K.load4(k0 + r, kvh, d, a, b, c, e);
                V.load4(k0 + r, kvh, d, va, vb, vc, ve);
            }
            sK[r][d / 2] = __floats2half2_rn(a, b);
            sK[r][d / 2 + 1] = __floats2half2_rn(c, e);
            sV[r][d / 2] = __floats2half2_rn(va, vb);
            sV[r][d / 2 + 1] = __floats2half2_rn(vc, ve);
        }
        __syncthreads();
        // scores: lane = key, rows w + 8 i
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int r = w + 8 * i, pos = p0 + r0 + r, key = k0 + lane;
            float s = 0.0f;
#pragma unroll 8
            for (int d2 = 0; d2 < kD / 2; ++d2) {
                const float2 kk = __half22float2(sK[lane][d2]);
                s += sQ[r][2 * d2] * kk.x + sQ[r][2 * d2 + 1] * kk.y;
            }
            const bool ok = r0 + r < T && key <= pos;
            s = ok ? s : -INFINITY;
            const float mt = wmax(s);
            const float mn = fmaxf(m[i], mt);
            const float p = ok ? __expf(s - mn) : 0.0f;
            const float corr = m[i] == -INFINITY ? 0.0f : __expf(m[i] - mn);
            l[i] = l[i] * corr + wsum(p);
            m[i] = mn;
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[i][j] *= corr;
            sP[r][lane] = p;
        }
        __syncwarp();
        // output: lane owns dims 4 lane .. 4 lane + 3 of the warp's rows
#pragma unroll 4
        for (int key = 0; key < kKT; ++key) {
            const float2 v01 = __half22float2(sV[key][2 * lane]), v23 = __half22float2(sV[key][2 * lane + 1]);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float p = sP[w + 8 * i][key];
                acc[i][0] += p * v01.x;
                acc[i][1] += p * v01.y;
                acc[i][2] += p * v23.x;
                acc[i][3] += p * v23.y;
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int r = r0 + w + 8 * i;
        if (r >= T) continue;
        const float inv = l[i] > 0.0f ? 1.0f / l[i] : 0.0f;
        __half* o = out + (size_t) r * old + (size_t) h * kD + 4 * lane;
        o[0] = __float2half(acc[i][0] * inv);
        o[1] = __float2half(acc[i][1] * inv);
        o[2] = __float2half(acc[i][2] * inv);
        o[3] = __float2half(acc[i][3] * inv);
    }
}

#if !defined(STRATA_USE_HIP)
// ---- the same on the tensor cores (sm_70+, FP16 in, F32 accumulate) in one pass with a lazily rescaled output (see
// below): S = Q K^T and O += P V per 64-key tile, the output in WMMA accumulators (registers).  Block (128 query rows, one query head), 8 warps
// of 16 rows, each with its Q in 8 fragments; 64-key tiles of K and V through shared memory, S and P per warp.
#ifndef STRATA_GQA_KTK
#define STRATA_GQA_KTK 64
#endif
constexpr int kTQ = 128, kTK = STRATA_GQA_KTK, kTW = 8;
constexpr int kKH = kTK / 2;   // a lane pair's keys of a tile
constexpr int kLdQ = kD + 8, kLdS = kTK + 4, kLdP = kTK + 8;   // (wmma: ldm % 8 halves / 4 floats)
constexpr size_t kTcTile = 2 * (size_t) kTK * kLdQ * 2 + (size_t) kTW * 16 * kLdS * 4 + (size_t) kTW * 16 * kLdP * 2;
constexpr size_t kTcSmem = kTcTile > (size_t) kTQ * kLdQ * 2 ? kTcTile : (size_t) kTQ * kLdQ * 2;   // (Q staging)
__global__ void __launch_bounds__(kTW * 32) gqa_attn_rows_tc_kernel(const float* __restrict__ q, int qld, GqaKv kv,
                                                                   int p0, int T, int n_head, float scale,
                                                                   __half* __restrict__ out, int old) {
    using namespace nvcuda;
    extern __shared__ __align__(32) unsigned char smem[];
    __half* sK = (__half*) smem;                               // [kTK][kLdQ]
    __half* sV = sK + kTK * kLdQ;                              // [kTK][kLdQ]
    float* sS = (float*) (sV + kTK * kLdQ);                    // [kTW][16][kLdS]
    __half* sP = (__half*) (sS + kTW * 16 * kLdS);             // [kTW][16][kLdP]
    const int h = blockIdx.y, r0 = blockIdx.x * kTQ, t = threadIdx.x, lane = t & 31, w = t >> 5;
    const int n_kv = kv.n_kv, kvh = h / (n_head / n_kv);
    __shared__ float scb[2][64];
    load_cbs(scb, kv.kk, kv.vk, t, kTW * 32);
    const Rows K = k_rows(kv, scb[0]), V = v_rows(kv, scb[1]);
    // Q (scaled) -> FP16 staging -> each warp's 8 fragments
    __half* sQ = (__half*) smem;
    for (int i = t; i < kTQ * kD; i += kTW * 32) {
        const int r = i / kD, d = i % kD;
        sQ[r * kLdQ + d] = __float2half(r0 + r < T ? q[(size_t) (r0 + r) * qld + (size_t) h * kD + d] * scale : 0.0f);
    }
    __syncthreads();
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fq[kD / 16];
#pragma unroll
    for (int k = 0; k < kD / 16; ++k) wmma::load_matrix_sync(fq[k], sQ + (w * 16) * kLdQ + k * 16, kLdQ);
    float* Sw = sS + w * 16 * kLdS;
    __half* Pw = sP + w * 16 * kLdP;
    // the lane pair (2 ri, 2 ri + 1) owns row ri of the warp's 16: keys kKH (lane & 1) .. + kKH - 1 of a tile
    const int ri = lane >> 1, half_i = lane & 1;
    const int row = r0 + w * 16 + ri, pos = p0 + row;
    const bool row_ok = row < T;
    const int last = p0 + min(r0 + kTQ, T) - 1;
    const int wlast = p0 + min(r0 + w * 16 + 16, T) - 1;       // the warp's own last position (fewer tiles above it)
    const auto load_tile = [&](int k0, bool with_v) {
        for (int i = t; i < kTK * (kD / 4); i += kTW * 32) {
            const int r = i / (kD / 4), d = (i % (kD / 4)) * 4;
            float a = 0.f, b = 0.f, c = 0.f, e = 0.f;
            if (k0 + r <= last) K.load4(k0 + r, kvh, d, a, b, c, e);
            __half2* kr = (__half2*) (sK + r * kLdQ + d);
            kr[0] = __floats2half2_rn(a, b);
            kr[1] = __floats2half2_rn(c, e);
            if (with_v) {
                if (k0 + r <= last) V.load4(k0 + r, kvh, d, a, b, c, e);
                __half2* vr = (__half2*) (sV + r * kLdQ + d);
                vr[0] = __floats2half2_rn(a, b);
                vr[1] = __floats2half2_rn(c, e);
            }
        }
    };
    const auto scores = [&]() {   // Sw (16 x kTK) = Q_w . K^T
#pragma unroll
        for (int n = 0; n < kTK / 16; ++n) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::fill_fragment(acc, 0.0f);
#pragma unroll
            for (int k = 0; k < kD / 16; ++k) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
                wmma::load_matrix_sync(fb, sK + (n * 16) * kLdQ + k * 16, kLdQ);
                wmma::mma_sync(acc, fq[k], fb, acc);
            }
            wmma::store_matrix_sync(Sw + n * 16, acc, kLdS, wmma::mem_row_major);
        }
        __syncwarp();
    };
    // ---- one pass: P = e^(s - m_ref) with a per-row reference max that moves only when a tile's max passes it by
    // more than kLazy (P <= e^kLazy fits FP16); then the accumulated output's rows are rescaled by e^(m_old - m_new),
    // a 16x16 fragment at a time through the warp's S area - a few times a row (its first tiles), so the output stays
    // in the WMMA accumulators and K / V are read once
    constexpr float kLazy = 8.0f;
    float m = -INFINITY, l = 0.0f;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fo[kD / 16];
#pragma unroll
    for (int n = 0; n < kD / 16; ++n) wmma::fill_fragment(fo[n], 0.0f);
    for (int k0 = 0; k0 <= last; k0 += kTK) {
        __syncthreads();
        load_tile(k0, true);
        __syncthreads();
        if (k0 > wlast) continue;   // (the warp's rows see none of these keys)
        scores();
        float sv[kKH], mt = -INFINITY;
#pragma unroll
        for (int j = 0; j < kKH; ++j) {
            const int c = half_i * kKH + j;
            sv[j] = row_ok && k0 + c <= pos ? Sw[ri * kLdS + c] : -INFINITY;
            mt = fmaxf(mt, sv[j]);
        }
        mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, 1));
        const bool need = mt > m + kLazy;   // (the first tile with keys: m = -inf)
        float corr = 1.0f;
        if (need) {
            corr = m == -INFINITY ? 0.0f : __expf(m - mt);
            l *= corr;
            m = mt;
        }
        float ps = 0.0f;
#pragma unroll
        for (int j = 0; j < kKH; ++j) {
            const float pj = sv[j] == -INFINITY ? 0.0f : __expf(sv[j] - m);
            ps += pj;
            Pw[ri * kLdP + half_i * kKH + j] = __float2half(pj);
        }
        l += ps + __shfl_xor_sync(0xffffffffu, ps, 1);
        __syncwarp();
        if (__any_sync(0xffffffffu, need)) {   // (S is spent: its area holds one fragment at a time)
#pragma unroll
            for (int n = 0; n < kD / 16; ++n) {
                wmma::store_matrix_sync(Sw, fo[n], kLdS, wmma::mem_row_major);
                __syncwarp();
#pragma unroll
                for (int j = 0; j < 8; ++j) Sw[ri * kLdS + half_i * 8 + j] *= corr;
                __syncwarp();
                wmma::load_matrix_sync(fo[n], Sw, kLdS, wmma::mem_row_major);
                __syncwarp();
            }
        }
#pragma unroll
        for (int k = 0; k < kTK / 16; ++k) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fp;
            wmma::load_matrix_sync(fp, Pw + k * 16, kLdP);
#pragma unroll
            for (int n = 0; n < kD / 16; ++n) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fv;
                wmma::load_matrix_sync(fv, sV + (k * 16) * kLdQ + n * 16, kLdQ);
                wmma::mma_sync(fo[n], fp, fv, fo[n]);
            }
        }
        __syncwarp();
    }
    const float inv_l = l > 0.0f ? 1.0f / l : 0.0f;
    // the output, 16 dims at a time through the warp's S area
#pragma unroll
    for (int n = 0; n < kD / 16; ++n) {
        wmma::store_matrix_sync(Sw, fo[n], kLdS, wmma::mem_row_major);
        __syncwarp();
        if (row_ok) {
            __half* o = out + (size_t) row * old + (size_t) h * kD + n * 16 + half_i * 8;
#pragma unroll
            for (int j = 0; j < 8; ++j) o[j] = __float2half(Sw[ri * kLdS + half_i * 8 + j] * inv_l);
        }
        __syncwarp();
    }
}
#endif

#if !defined(STRATA_USE_HIP)
// ---- decode on the tensor cores: block (K/V head, 256-key split), 4 warps of 64 keys in 16-key tiles.  The G query
// heads of the K/V head are the rows of one 16-row WMMA operand (rows G.. zero), so a tile's scores for all of them are
// 8 products instead of G warp reductions a key; the online softmax is the prompt kernel's lazy one.  The warps' (m, l,
// O) are merged in shared memory into the split's part, the layout gqa_combine_kernel reads.
constexpr int kDW = 4, kDT = 16;                                   // warps, keys a tile
constexpr int kDLdS = kDT + 4, kDLdP = kDT + 8, kDLdO = kD + 4;    // (wmma: ldm % 8 halves / 4 floats)
__global__ void __launch_bounds__(kDW * 32) gqa_attn_tc_kernel(const float* __restrict__ q, GqaKv kv, int n_pos,
                                                               int n_head, float scale, float* __restrict__ part) {
    using namespace nvcuda;
    __shared__ __align__(32) __half sQ[16 * kLdQ];
    __shared__ __align__(32) __half sKV[kDW][2 * kDT * kLdQ];      // per warp: K tile, V tile (then its O, F32)
    __shared__ __align__(32) float sS[kDW][16 * kDLdS];
    __shared__ __align__(32) __half sP[kDW][16 * kDLdP];
    __shared__ float sm[kDW][16], sl[kDW][16];
    const int n_kv = kv.n_kv, kvh = blockIdx.x, split = blockIdx.y, G = n_head / n_kv;
    const int t = threadIdx.x, lane = t & 31, w = t >> 5;
    const Keys ks = keys_of(kv, n_pos);
    const bool cached = kv.sel != nullptr;
    for (int i = t; i < 16 * kD; i += kDW * 32) {
        const int r = i / kD, d = i % kD;
        sQ[r * kLdQ + d] = __float2half(r < G ? q[(size_t) (kvh * G + r) * kD + d] * scale : 0.0f);
    }
    __syncthreads();
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fq[kD / 16];
#pragma unroll
    for (int k = 0; k < kD / 16; ++k) wmma::load_matrix_sync(fq[k], sQ + k * 16, kLdQ);
    // the codebooks in sQ's memory, Q being in the fragments now (a table of its own took the block past half the SM's
    // shared memory: one block an SM instead of two, the kernel ~half as fast)
    __syncthreads();
    float (*scb)[64] = (float (*)[64]) (void*) sQ;
    load_cbs(scb, kv.kk, kv.vk, t, kDW * 32);
    __syncthreads();
    const Rows K = k_rows(kv, scb[0], cached), V = v_rows(kv, scb[1], cached);
    __half* Kw = sKV[w];
    __half* Vw = sKV[w] + kDT * kLdQ;
    float* Sw = sS[w];
    __half* Pw = sP[w];
    const int ri = lane >> 1, half_i = lane & 1;                  // the lane pair of row (head) ri: keys 8 half_i ..
    const bool row_ok = ri < G;
    const int k_lo = split * kChunk + w * (kChunk / kDW), k_hi = min(k_lo + kChunk / kDW, ks.n_idx);
    constexpr float kLazy = 8.0f;
    float m = -INFINITY, l = 0.0f;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fo[kD / 16];
#pragma unroll
    for (int n = 0; n < kD / 16; ++n) wmma::fill_fragment(fo[n], 0.0f);
    for (int k0 = k_lo; k0 < k_hi; k0 += kDT) {
        // the tile's K and V rows (16 rows x 32 groups of 4: 16 per lane)
#pragma unroll 16   // (all 16 rows' loads in flight: ~2% over 4)
        for (int i = lane; i < kDT * (kD / 4); i += 32) {
            const int r = i / (kD / 4), d = (i % (kD / 4)) * 4;
            float a = 0.f, b = 0.f, c = 0.f, e = 0.f, va = 0.f, vb = 0.f, vc = 0.f, ve = 0.f;
            if (k0 + r < k_hi && ks.ok(k0 + r)) {
                const long long pos = ks.pos(k0 + r);
                K.load4(pos, kvh, d, a, b, c, e);
                V.load4(pos, kvh, d, va, vb, vc, ve);
            }
            __half2* kr = (__half2*) (Kw + r * kLdQ + d);
            __half2* vr = (__half2*) (Vw + r * kLdQ + d);
            kr[0] = __floats2half2_rn(a, b);
            kr[1] = __floats2half2_rn(c, e);
            vr[0] = __floats2half2_rn(va, vb);
            vr[1] = __floats2half2_rn(vc, ve);
        }
        __syncwarp();
        {   // S (16 x 16) = Q . K^T
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::fill_fragment(acc, 0.0f);
#pragma unroll
            for (int k = 0; k < kD / 16; ++k) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
                wmma::load_matrix_sync(fb, Kw + k * 16, kLdQ);
                wmma::mma_sync(acc, fq[k], fb, acc);
            }
            wmma::store_matrix_sync(Sw, acc, kDLdS, wmma::mem_row_major);
        }
        __syncwarp();
        float sv[8], mt = -INFINITY;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int c = half_i * 8 + j;
            sv[j] = row_ok && k0 + c < k_hi && ks.ok(k0 + c) ? Sw[ri * kDLdS + c] : -INFINITY;
            mt = fmaxf(mt, sv[j]);
        }
        mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, 1));
        const bool need = mt > m + kLazy;
        float corr = 1.0f;
        if (need) {
            corr = m == -INFINITY ? 0.0f : __expf(m - mt);
            l *= corr;
            m = mt;
        }
        float ps = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float pj = sv[j] == -INFINITY ? 0.0f : __expf(sv[j] - m);
            ps += pj;
            Pw[ri * kDLdP + half_i * 8 + j] = __float2half(pj);
        }
        l += ps + __shfl_xor_sync(0xffffffffu, ps, 1);
        __syncwarp();
        if (__any_sync(0xffffffffu, need)) {
#pragma unroll
            for (int n = 0; n < kD / 16; ++n) {
                wmma::store_matrix_sync(Sw, fo[n], kDLdS, wmma::mem_row_major);
                __syncwarp();
#pragma unroll
                for (int j = 0; j < 8; ++j) Sw[ri * kDLdS + half_i * 8 + j] *= corr;
                __syncwarp();
                wmma::load_matrix_sync(fo[n], Sw, kDLdS, wmma::mem_row_major);
                __syncwarp();
            }
        }
        {   // O (16 x 128) += P (16 x 16) . V (16 x 128)
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fp;
            wmma::load_matrix_sync(fp, Pw, kDLdP);
#pragma unroll
            for (int n = 0; n < kD / 16; ++n) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fv;
                wmma::load_matrix_sync(fv, Vw + n * 16, kLdQ);
                wmma::mma_sync(fo[n], fp, fv, fo[n]);
            }
        }
        __syncwarp();
    }
    // the warp's O (relative to its m) into its K / V area as F32, its m and l beside
    float* Ow = (float*) sKV[w];
#pragma unroll
    for (int n = 0; n < kD / 16; ++n) wmma::store_matrix_sync(Ow + n * 16, fo[n], kDLdO, wmma::mem_row_major);
    if (half_i == 0) {
        sm[w][ri] = m;
        sl[w][ri] = l;
    }
    __syncthreads();
    // merge the warps: thread t = dim t, every head of the group
    for (int g = 0; g < G; ++g) {
        float M = -INFINITY;
#pragma unroll
        for (int x = 0; x < kDW; ++x) M = fmaxf(M, sm[x][g]);
        float acc = 0.0f, L = 0.0f;
#pragma unroll
        for (int x = 0; x < kDW; ++x) {
            const float f = sm[x][g] == -INFINITY ? 0.0f : __expf(sm[x][g] - M);
            acc += f * ((const float*) sKV[x])[g * kDLdO + t];
            L += f * sl[x][g];
        }
        float* o = part + ((size_t) (kvh * G + g) * gridDim.y + split) * (kD + 2);
        o[t] = acc;
        if (t == 0) {
            o[kD] = M;
            o[kD + 1] = L;
        }
    }
}
#endif

// ---- KV streaming (glm_fast.hpp GqaStream)

// a page's summary: grid (pages, n_kv), kD threads (thread = dimension): the min and max of the K rows of positions
// [page * kGqaPage, min(page end, pos_hi)) from src (a plain view)
__global__ void __launch_bounds__(kD) gqa_summ_kernel(GqaKv src, GqaStream st, int page_lo, int pos_hi) {
    __shared__ float scb[2][64];
    const int page = page_lo + blockIdx.x, kvh = blockIdx.y, t = threadIdx.x;
    load_cbs(scb, src.kk, src.vk, t, kD);
    __syncthreads();
    const Rows K = k_rows(src, scb[0]);
    const int lo = page * kGqaPage, hi = min(lo + kGqaPage, pos_hi);
    float mn = INFINITY, mx = -INFINITY;
    for (int pos = lo; pos < hi; ++pos) {
        float x[4];
        K.load4(pos, kvh, t & ~3, x[0], x[1], x[2], x[3]);
        mn = fminf(mn, x[t & 3]);
        mx = fmaxf(mx, x[t & 3]);
    }
    __half* o = (__half*) st.summ + ((size_t) page * src.n_kv + kvh) * 2 * kD;
    o[t] = __float2half(hi > lo ? mn : 0.0f);
    o[kD + t] = __float2half(hi > lo ? mx : 0.0f);
}

// ... the decode's one new position: its row folded into its page's min / max (the page's first position starts them)
// - O(1) where the kernel above re-reads the whole page (a 64-position page: ~50 us a layer, 4 ms a token on 80 layers)
__global__ void __launch_bounds__(kD) gqa_summ_one_kernel(GqaKv src, GqaStream st, int pos) {
    __shared__ float scb[2][64];
    const int kvh = blockIdx.x, t = threadIdx.x;
    load_cbs(scb, src.kk, src.vk, t, kD);
    __syncthreads();
    const Rows K = k_rows(src, scb[0]);
    float x[4];
    K.load4(pos, kvh, t & ~3, x[0], x[1], x[2], x[3]);
    const float v = x[t & 3];
    __half* o = (__half*) st.summ + ((size_t) (pos / kGqaPage) * src.n_kv + kvh) * 2 * kD;
    const bool first = pos % kGqaPage == 0;
    o[t] = __float2half(first ? v : fminf(__half2float(o[t]), v));
    o[kD + t] = __float2half(first ? v : fmaxf(__half2float(o[kD + t]), v));
}

// every query head's bound per page: grid (n_kv, pages / 128), 128 threads (thread = page) -> ub[head][page]
__global__ void __launch_bounds__(128) gqa_sel_ub_kernel(const float* __restrict__ q, int n_head, int n_kv,
                                                         GqaStream st, int np, float* __restrict__ ub) {
    __shared__ float sq[kMaxGroup][kD];
    const int kvh = blockIdx.x, G = n_head / n_kv, t = threadIdx.x, page = blockIdx.y * 128 + t;
    for (int i = t; i < G * kD; i += 128) sq[i / kD][i % kD] = q[(size_t) kvh * G * kD + i];
    __syncthreads();
    if (page >= np) return;
    const __half2* sm = (const __half2*) ((const __half*) st.summ + ((size_t) page * n_kv + kvh) * 2 * kD);
    float acc[kMaxGroup];
#pragma unroll
    for (int g = 0; g < kMaxGroup; ++g) acc[g] = 0.0f;
    for (int d2 = 0; d2 < kD / 2; ++d2) {
        const float2 mn = __half22float2(sm[d2]), mx = __half22float2(sm[kD / 2 + d2]);
#pragma unroll
        for (int g = 0; g < kMaxGroup; ++g) {
            if (g >= G) break;
            const float q0 = sq[g][2 * d2], q1 = sq[g][2 * d2 + 1];
            acc[g] += fmaxf(q0 * mn.x, q0 * mx.x) + fmaxf(q1 * mn.y, q1 * mx.y);
        }
    }
#pragma unroll
    for (int g = 0; g < kMaxGroup; ++g)
        if (g < G) ub[(size_t) (kvh * G + g) * np + page] = acc[g];
}

// one block of 1024: a page's score = the max over the heads of (its bound - the head's best bound); the first sink and
// the last recent pages first, then the budget best (a bisection for the threshold) -> sel / nsel
constexpr int kSelT = 1024;
__global__ void __launch_bounds__(kSelT) gqa_sel_pick_kernel(const float* __restrict__ ub, int n_head, int np,
                                                             int budget, int sink, int recent, int* __restrict__ sel,
                                                             int* __restrict__ nsel, int max_sel) {
    extern __shared__ float sc[];   // [np]
    __shared__ float hm[64];
    __shared__ int cnt;
    const int t = threadIdx.x, lane = t & 31, w = t >> 5;
    const int total = min(np, sink + recent + budget);
    if (total >= np) {   // every page
        for (int p = t; p < np; p += kSelT) sel[p] = p;
        if (t == 0) *nsel = np;
        return;
    }
    for (int h = w; h < n_head; h += kSelT / 32) {
        float m = -INFINITY;
        for (int p = lane; p < np; p += 32) m = fmaxf(m, ub[(size_t) h * np + p]);
        m = wmax(m);
        if (lane == 0) hm[h & 63] = m;
    }
    __syncthreads();
    float lo = 0.0f;
    for (int p = t; p < np; p += kSelT) {
        float v;
        if (p < sink || p >= np - recent) {
            v = 3.0e38f;
        } else {
            v = -INFINITY;
            for (int h = 0; h < n_head; ++h) v = fmaxf(v, ub[(size_t) h * np + p] - hm[h & 63]);
        }
        sc[p] = v;
        lo = fminf(lo, v);
    }
    // the total best pages: page p is taken when fewer than total pages rank above it (higher score, or the same score
    // at a lower index) - each thread counts for its pages over the scores in shared memory (np <= ~2K at 128K
    // context: a few us; the 40-step bisection it replaces took ~70 us a layer, a single-thread radix scan more)
    if (t == 0) cnt = 0;
    __syncthreads();
    for (int p = t; p < np; p += kSelT) {
        const float v = sc[p];
        int above = 0;
        for (int q2 = 0; q2 < np && above < total; ++q2) {
            const float u = sc[q2];
            above += u > v || (u == v && q2 < p);
        }
        if (above < total) {
            const int k = atomicAdd(&cnt, 1);
            if (k < max_sel) sel[k] = p;
        }
    }
    (void) lo;
    __syncthreads();
    if (t == 0) *nsel = min(cnt, max_sel);
}

// ---- the selection as three short kernels (the two above, one block for the pick, took 85 us a layer at 406 pages and
// 340 at 1460 - 7-27 ms a token on 80 layers): reset, the bounds with each head's best, the pick over many blocks
__device__ __forceinline__ unsigned int sel_key(float f) {
    const unsigned int u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__device__ __forceinline__ float sel_val(unsigned int k) {
    return __uint_as_float((k & 0x80000000u) ? (k & 0x7fffffffu) : ~k);
}
__global__ void gqa_sel_init_kernel(unsigned int* hm, int* nsel, int n_head) {
    if ((int) threadIdx.x < n_head) hm[threadIdx.x] = 0u;   // (below every float's key)
    if (threadIdx.x == 0) *nsel = 0;
}
// the bounds: grid (n_kv, pages / 32), 32 G threads (thread: page t / G of the block's 32, head t % G); the 32 pages'
// summaries staged in shared memory (coalesced); ub[head][page], and each head's best into hm
__global__ void __launch_bounds__(32 * kMaxGroup) gqa_sel_ub2_kernel(const float* __restrict__ q, int n_head, int n_kv,
                                                                    GqaStream st, int np, float* __restrict__ ub,
                                                                    unsigned int* __restrict__ hm) {
    __shared__ float sq[kMaxGroup][kD + 1];   // (+1: a warp's 8 heads read the same dimension in 8 banks)
    __shared__ __half ss[32][2 * kD + 2];     // (+1 word: the 4 pages a warp reads at once fall in different banks)
    __shared__ float sbest[32 * kMaxGroup / 32][kMaxGroup];
    const int kvh = blockIdx.x, G = n_head / n_kv, t = threadIdx.x, nt = 32 * G, p0 = blockIdx.y * 32;
    for (int i = t; i < G * kD; i += nt) sq[i / kD][i % kD] = q[(size_t) kvh * G * kD + i];
    for (int i = t; i < 32 * kD; i += nt) {   // (half2 units: 32 pages x 128)
        const int pp = i / kD, c = i % kD;
        __half2 v = __floats2half2_rn(0.0f, 0.0f);
        if (p0 + pp < np) v = ((const __half2*) st.summ)[((size_t) (p0 + pp) * n_kv + kvh) * kD + c];
        *(__half2*) &ss[pp][2 * c] = v;
    }
    __syncthreads();
    const int pp = t / G, g = t % G, page = p0 + pp;
    float acc = 0.0f;
    const __half2* sm = (const __half2*) ss[pp];
#pragma unroll 4
    for (int d2 = 0; d2 < kD / 2; ++d2) {
        const float2 mn = __half22float2(sm[d2]), mx = __half22float2(sm[kD / 2 + d2]);
        const float q0 = sq[g][2 * d2], q1 = sq[g][2 * d2 + 1];
        acc += fmaxf(q0 * mn.x, q0 * mx.x) + fmaxf(q1 * mn.y, q1 * mx.y);
    }
    if (page < np) ub[(size_t) (kvh * G + g) * np + page] = acc;
    // each head's best of the block's pages: over the warp's pages (lanes g, g + G, ...), then the warps, one atomic
    // a head (the atomics per page and head serialised on 64 addresses: ~20 us of this kernel)
    float best = page < np ? acc : -INFINITY;
    for (int o = G; o < 32; o <<= 1) best = fmaxf(best, __shfl_xor_sync(0xffffffffu, best, o));
    if ((t & 31) < G) sbest[t >> 5][g] = best;
    __syncthreads();
    if (t < G) {
        float m = -INFINITY;
        for (int w = 0; w < nt / 32; ++w) m = fmaxf(m, sbest[w][t]);
        atomicMax(&hm[kvh * G + t], sel_key(m));
    }
}
// the pages' scores: the max over the heads of the bound less the head's best (the first sink and the last recent pages
// forced) - grid pages / 256
__global__ void __launch_bounds__(256) gqa_sel_score_kernel(const float* __restrict__ ub,
                                                            const unsigned int* __restrict__ hm, int n_head, int np,
                                                            int sink, int recent, float* __restrict__ sc) {
    __shared__ float shm[64];
    const int t = threadIdx.x, p = blockIdx.x * 256 + t;
    if (t < n_head && t < 64) shm[t] = sel_val(hm[t]);
    __syncthreads();
    if (p >= np) return;
    float v;
    if (p < sink || p >= np - recent) {
        v = 3.0e38f;
    } else {
        v = -INFINITY;
        for (int h = 0; h < n_head; ++h) v = fmaxf(v, ub[(size_t) h * np + p] - shm[h & 63]);
    }
    sc[p] = v;
}
// the pick: grid pages / 32, 256 threads - every block reads all the scores into shared memory, ranks its 32 pages
// (eight threads a page) and appends those that fewer than `total` pages outrank (higher score, or equal at a lower
// index)
__global__ void __launch_bounds__(256) gqa_sel_pick2_kernel(const float* __restrict__ scg, int np, int total,
                                                            int* __restrict__ sel, int* __restrict__ nsel, int max_sel) {
    extern __shared__ float sc[];   // [np]
    const int t = threadIdx.x;
    for (int p = t; p < np; p += 256) sc[p] = scg[p];
    __syncthreads();
    const int p = blockIdx.x * 32 + (t >> 3), part = t & 7;
    int above = 0;
    if (p < np) {
        const float v = sc[p];
        for (int q2 = part; q2 < np; q2 += 8) {
            const float u = sc[q2];
            above += u > v || (u == v && q2 < p);
        }
    }
    above += __shfl_xor_sync(0xffffffffu, above, 1);
    above += __shfl_xor_sync(0xffffffffu, above, 2);
    above += __shfl_xor_sync(0xffffffffu, above, 4);
    if (p < np && part == 0 && above < total) {
        const int k = atomicAdd(nsel, 1);
        if (k < max_sel) sel[k] = p;
    }
}
__global__ void gqa_sel_all_kernel(int np, int* sel, int* nsel) {
    for (int p = threadIdx.x; p < np; p += blockDim.x) sel[p] = p;
    if (threadIdx.x == 0) *nsel = np;
}

// one block of 1024: the selected pages already cached are marked used by this step (and referenced); the others are
// claimed CLOCK's victims (never a slot this step uses) -> miss / mslot / nmiss for the copy below
__global__ void __launch_bounds__(1024) gqa_resolve_kernel(GqaStream st, GqaSelBufs b) {
    // the slots' tables in shared memory while CLOCK sweeps them (thread 0's steps were global round trips: ~50 us a
    // layer); written back at the end
    extern __shared__ int sm_tab[];   // [3][R]: epoch, reference bit, page
    int* s_epoch = sm_tab;
    int* s_ref = sm_tab + st.R;
    int* s_page = sm_tab + 2 * st.R;
    __shared__ int s_ep, s_nm;
    const int t = threadIdx.x;
    for (int i = t; i < st.R; i += 1024) {
        s_epoch[i] = st.sepoch[i];
        s_ref[i] = st.sref[i];
        s_page[i] = st.spage[i];
    }
    if (t == 0) {
        s_ep = ++st.ctl[1];
        s_nm = 0;
    }
    __syncthreads();
    const int ep = s_ep, ns = *b.nsel;
    for (int i = t; i < ns; i += 1024) {
        const int p = b.sel[i], sl = st.ptab[p];
        if (sl >= 0) {
            s_epoch[sl] = ep;
            s_ref[sl] = 1;
        } else {
            b.miss[atomicAdd(&s_nm, 1)] = p;
        }
    }
    __syncthreads();
    if (t == 0) {
        int hand = st.ctl[0], k = 0;
        for (; k < s_nm; ++k) {
            int sl = -1;
            for (int step = 0; step < 4 * st.R; ++step) {
                const int c = hand;
                hand = hand + 1 == st.R ? 0 : hand + 1;
                if (s_epoch[c] == ep) continue;
                if (s_ref[c]) {
                    s_ref[c] = 0;
                    continue;
                }
                sl = c;
                break;
            }
            if (sl < 0) break;   // (cannot happen: R exceeds the largest selection)
            const int old = s_page[sl];
            if (old >= 0) st.ptab[old] = -1;
            const int p = b.miss[k];
            s_page[sl] = p;
            st.ptab[p] = sl;
            s_epoch[sl] = ep;
            s_ref[sl] = 1;
            b.mslot[k] = sl;
        }
        st.ctl[0] = hand;
        *b.nmiss = k;
    }
    __syncthreads();
    for (int i = t; i < st.R; i += 1024) {
        st.sepoch[i] = s_epoch[i];
        st.sref[i] = s_ref[i];
        st.spage[i] = s_page[i];
    }
}

// grid max_sel, 256 threads: block k copies miss k's page (kGqaPage records of slice bytes) from host memory
__global__ void __launch_bounds__(256) gqa_page_copy_kernel(GqaStream st, GqaSelBufs b, int slice) {
    const int k = blockIdx.x;
    if (k >= *b.nmiss) return;
    const size_t bytes = (size_t) kGqaPage * slice;
    const uint4* src = (const uint4*) (st.host + (size_t) b.miss[k] * bytes);
    uint4* dst = (uint4*) (st.slots + (size_t) b.mslot[k] * bytes);
    for (size_t i = threadIdx.x; i < bytes / 16; i += 256) dst[i] = src[i];
}

__global__ void gqa_invalidate_kernel(GqaStream st, int page_lo, int page_hi) {
    const int p = page_lo + blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= page_hi) return;
    const int sl = st.ptab[p];
    if (sl >= 0) {
        st.spage[sl] = -1;
        st.ptab[p] = -1;
    }
}

// ---- decode, wide (the default): block (K/V head, split of `chunk` keys), 128 threads, ~8 KB of shared memory - many
// blocks an SM, so many loads in flight (the tensor-core kernel below fits two: ~10% of the memory's bandwidth).  Per
// 128-key tile: (1) thread t takes key t's scores for the G heads, streaming its K row against q in shared memory (a
// broadcast read); (2) warp w the online softmax of heads w, w + 4, ...; (3) warp w the keys 32 w .. 32 w + 31 of P.V,
// a lane 4 dimensions, every head's output in registers.  The warps' outputs are summed once, at the end.
constexpr int kST = 128;
// q (G heads, shared memory) . head h's K row in the record at bp -> sc[G], the row read in 16-byte loads and decoded in
// registers (a thread a key: dims at a time from one load, nothing for L1 to keep between them)
// - TPK threads a key: this one's part (0 .. TPK - 1) of the dims, its partial sums
template <int G, int TPK>
__device__ __forceinline__ void dot_row(const Rows& K, const uint8_t* bp, int h, const float (*sq)[kD], float* sc,
                                        int part) {
    const uint8_t* cp = K.codes(bp, h);
    const uint8_t* spp = K.scales(bp, h);
    const auto fma8 = [&](int d, const float* x) {
#pragma unroll
        for (int g = 0; g < G; ++g) {
            const float4 a = *(const float4*) &sq[g][d], b = *(const float4*) &sq[g][d + 4];
            sc[g] += a.x * x[0] + a.y * x[1] + a.z * x[2] + a.w * x[3] + b.x * x[4] + b.y * x[5] + b.z * x[6] + b.w * x[7];
        }
    };
    float x[8];
    if (K.kind == kF16) {
#pragma unroll 4
        for (int c = part * (kD / 8 / TPK); c < (part + 1) * (kD / 8 / TPK); ++c) {
            const uint4 u = ((const uint4*) cp)[c];
            const __half2* hh = (const __half2*) &u;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 f = __half22float2(hh[j]);
                x[2 * j] = f.x;
                x[2 * j + 1] = f.y;
            }
            fma8(8 * c, x);
        }
    } else if (K.kind == kI8) {
        const uint2 s4 = *(const uint2*) spp;   // (4 FP16 scales, one a 32)
        const __half* hs = (const __half*) &s4;
#pragma unroll
        for (int c = part * (kD / 16 / TPK); c < (part + 1) * (kD / 16 / TPK); ++c) {
            const uint4 u = ((const uint4*) cp)[c];
            const float scb = __half2float(hs[c / 2]);
            const int8_t* qb = (const int8_t*) &u;
#pragma unroll
            for (int k = 0; k < 2; ++k) {
#pragma unroll
                for (int j = 0; j < 8; ++j) x[j] = (float) qb[8 * k + j] * scb;
                fma8(16 * c + 8 * k, x);
            }
        }
    } else if (K.kind == kT4 || K.kind == kT6) {
        const float rs = *(const float*) spp;
#pragma unroll
        for (int c = part * (kD / 32 / TPK); c < (part + 1) * (kD / 32 / TPK); ++c) {
            const uint4 lo = ((const uint4*) cp)[c];   // 32 nibbles: dims 32 c ..
            uint2 hi = make_uint2(0, 0);
            if (K.kind == kT6) hi = ((const uint2*) (cp + kD / 2))[c];   // their high 2 bits
            const unsigned lw[4] = {lo.x, lo.y, lo.z, lo.w};
            const unsigned hw[4] = {hi.x & 0xffffu, hi.x >> 16, hi.y & 0xffffu, hi.y >> 16};
#pragma unroll
            for (int k = 0; k < 4; ++k) {
#pragma unroll
                for (int j = 0; j < 8; ++j) x[j] = K.lv[((lw[k] >> (4 * j)) & 15u) | (((hw[k] >> (2 * j)) & 3u) << 4)];
                fma8(32 * c + 8 * k, x);
            }
        }
#pragma unroll
        for (int g = 0; g < G; ++g) sc[g] *= rs;
    } else {   // (3-bit and 4-bit linear K: no packing of their own here)
#pragma unroll 4
        for (int d = part * (kD / TPK); d < (part + 1) * (kD / TPK); d += 4) {
            K.load4_b(bp, h, d, x[0], x[1], x[2], x[3]);
#pragma unroll
            for (int g = 0; g < G; ++g) {
                const float4 a = *(const float4*) &sq[g][d];
                sc[g] += a.x * x[0] + a.y * x[1] + a.z * x[2] + a.w * x[3];
            }
        }
    }
}
template <int G, int TK, bool STAGE>
__global__ void __launch_bounds__(kST) gqa_attn_wide_kernel(const float* __restrict__ q, GqaKv kv, int n_pos,
                                                            float scale, int chunk, int vstride,
                                                            float* __restrict__ part) {
    extern __shared__ __align__(16) uint8_t sv[];   // vstride > 0: the tile's V rows (codes, scales) a key a row
    constexpr int NW = kST / 32, HW = (G + NW - 1) / NW;   // warps, heads a warp
    constexpr int TPK = kST / TK, KW = TK / NW;            // threads a key (phase 1), keys a warp (phase 3)
    constexpr int SP = TK + 1, SPN = G * SP > G * kD ? G * SP : G * kD;
    __shared__ __align__(16) float sq[G][kD];
    __shared__ float spm[SPN];   // the tile's scores / probabilities [G][TK + 1]; at the end the warps' sum [G][kD]
    __shared__ float scorr[G], smx[G], sls[G];
    __shared__ float scb[2][64];
    const int kvh = blockIdx.x, split = blockIdx.y, t = threadIdx.x, lane = t & 31, w = t >> 5;
    load_cbs(scb, kv.kk, kv.vk, t, kST);
    for (int i = t; i < G * kD; i += kST) sq[i / kD][i % kD] = q[(size_t) kvh * G * kD + i] * scale;
    __syncthreads();
    const Keys ks = keys_of(kv, n_pos);
    const bool cached = kv.sel != nullptr;
    const Rows K = k_rows(kv, scb[0], cached), V = v_rows(kv, scb[1], cached);
    const int k_lo = split * chunk, k_hi = min(k_lo + chunk, ks.n_idx);
    float m_r[HW], l_r[HW];
#pragma unroll
    for (int j = 0; j < HW; ++j) {
        m_r[j] = -INFINITY;
        l_r[j] = 0.0f;
    }
    float acc[G][4];
#pragma unroll
    for (int g = 0; g < G; ++g) acc[g][0] = acc[g][1] = acc[g][2] = acc[g][3] = 0.0f;
    for (int k0 = k_lo; k0 < k_hi; k0 += TK) {
        {   // (1) key k0 + t / TPK, its part t % TPK of the dims (the parts' sums: neighbouring lanes)
            const int kt = t / TPK, i = k0 + kt;
            float sc[G];
            const bool ok = i < k_hi && ks.ok(i);
#pragma unroll
            for (int g = 0; g < G; ++g) sc[g] = 0.0f;
            if (ok) dot_row<G, TPK>(K, K.base(ks.pos(i)), kvh, sq, sc, t % TPK);
#pragma unroll
            for (int o = 1; o < TPK; o *= 2)
#pragma unroll
                for (int g = 0; g < G; ++g) sc[g] += __shfl_xor_sync(0xffffffffu, sc[g], o);
            if (t % TPK == 0)
#pragma unroll
                for (int g = 0; g < G; ++g) spm[g * SP + kt] = ok ? sc[g] : -INFINITY;
        }
        if (STAGE) {
            // the tile's V rows staged: key kt's codes and scales copied by its TPK threads (16-byte loads, all the
            // block's in flight at once - a lane's 2-8 bytes a key, one key after another, was latency bound); a masked
            // key's row is zeros (its probability is 0, its row may be stale)
            const int kt = t / TPK, i = k0 + kt, nc = V.cb / 16;
            uint4* dst = (uint4*) (sv + (size_t) kt * vstride);
            if (i < k_hi && ks.ok(i)) {
                const uint8_t* bp = V.base(ks.pos(i));
                const uint4* src = (const uint4*) V.codes(bp, kvh);
                for (int c = t % TPK; c < nc; c += TPK) dst[c] = src[c];
                if (t % TPK == 0) {
                    const uint8_t* ss = V.scales(bp, kvh);
                    if (V.sb == 8) *(uint2*) (sv + (size_t) kt * vstride + V.cb) = *(const uint2*) ss;
                    else *(unsigned*) (sv + (size_t) kt * vstride + V.cb) = *(const unsigned*) ss;
                }
            } else {
                for (int c = t % TPK; c < nc; c += TPK) dst[c] = make_uint4(0, 0, 0, 0);
                if (t % TPK == 0) *(uint2*) (sv + (size_t) kt * vstride + V.cb) = make_uint2(0, 0);
            }
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < HW; ++j) {   // (2)
            const int g = w + NW * j;
            if (g >= G) break;
            float* sp = spm + g * SP;
            float mt = -INFINITY;
            for (int x = lane; x < TK; x += 32) mt = fmaxf(mt, sp[x]);
            mt = wmax(mt);
            const float mn = fmaxf(m_r[j], mt);
            const float corr = m_r[j] == -INFINITY ? 0.0f : __expf(m_r[j] - mn);
            float ls = 0.0f;
            for (int x = lane; x < TK; x += 32) {
                const float v = sp[x];
                const float pv = v == -INFINITY ? 0.0f : __expf(v - mn);
                sp[x] = pv;
                ls += pv;
            }
            ls = wsum(ls);
            l_r[j] = l_r[j] * corr + ls;
            m_r[j] = mn;
            if (lane == 0) scorr[g] = corr;
        }
        __syncthreads();
        {   // (3) warp w the keys KW w .. KW w + KW - 1, a lane 4 dims of every head
#pragma unroll
            for (int g = 0; g < G; ++g) {
                const float c = scorr[g];
                acc[g][0] *= c; acc[g][1] *= c; acc[g][2] *= c; acc[g][3] *= c;
            }
            const int d4 = 4 * lane;
            if (STAGE) {
#pragma unroll 8
                for (int x = 0; x < KW; ++x) {
                    const int kk = w * KW + x;
                    float v0, v1, v2, v3;
                    V.load4_s(sv + (size_t) kk * vstride, d4, v0, v1, v2, v3);
#pragma unroll
                    for (int g = 0; g < G; ++g) {
                        const float pv = spm[g * SP + kk];
                        acc[g][0] += pv * v0; acc[g][1] += pv * v1; acc[g][2] += pv * v2; acc[g][3] += pv * v3;
                    }
                }
            } else
            // (FP16 V, whose rows are too large to stage: a lane its 4 values, a key at a time.  A no-test loop over the
            // valid prefix where the view is not streamed: 143 registers, 25% slower)
#pragma unroll 8
            for (int x = 0; x < KW; ++x) {
                const int kk = w * KW + x, i = k0 + kk;
                if (!(i < k_hi && ks.ok(i))) continue;   // (masked keys: their rows may be stale)
                float v0, v1, v2, v3;
                V.load4(ks.pos(i), kvh, d4, v0, v1, v2, v3);
#pragma unroll
                for (int g = 0; g < G; ++g) {
                    const float pv = spm[g * SP + kk];
                    acc[g][0] += pv * v0; acc[g][1] += pv * v1; acc[g][2] += pv * v2; acc[g][3] += pv * v3;
                }
            }
        }
        __syncthreads();   // (spm, sv: the next tile's)
    }
    // the warps' outputs (all relative to the same running max) summed in spm, one warp after another
    float* sO = spm;
#pragma unroll
    for (int x = 0; x < NW; ++x) {
        if (w == x)
#pragma unroll
            for (int g = 0; g < G; ++g)
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    float& o = sO[g * kD + 4 * lane + c];
                    o = (x == 0 ? 0.0f : o) + acc[g][c];
                }
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < HW; ++j) {
        const int g = w + NW * j;
        if (g < G && lane == 0) {
            smx[g] = m_r[j];
            sls[g] = l_r[j];
        }
    }
    __syncthreads();
    for (int g = 0; g < G; ++g) {
        float* o = part + ((size_t) (kvh * G + g) * gridDim.y + split) * (kD + 2);
        o[t] = sO[g * kD + t];
        if (t == 0) {
            o[kD] = smx[g];
            o[kD + 1] = sls[g];
        }
    }
}

#if !defined(STRATA_USE_HIP)
// ---- decode on the tensor cores (sm_70+, WMMA m8n32k16: the G = 8 query heads of a K/V head are the 8 rows).  A
// 128-key tile: (1) warp w the scores of keys 32 w .. 32 w + 31 - each lane decodes its key's row to FP16, 32 dims at a
// time, into the warp's buffer, then Q . K^T on the tensor cores (F32 sums); (2) the online softmax (warp w heads w,
// w + 4), the probabilities as FP16; (3) warp w output dims 32 w .. 32 w + 31 of P . V - each lane decodes its key's 32
// dims of V, 32 keys at a time.  The warps split the dims in (3): no merge across warps; thread t keeps dim t of every
// head's output.  On a K8V4 / TQ6V4 cache at 26K positions 151-166 against 225-230 us a layer for the wide kernel.
// 32 dims (32 c .. 32 c + 31) of a row: its bytes (ld32), then FP16 (dec32: 16 half2, dims in order).  KIND a
// compile-time constant.  (A K row's 4 chunks loaded at once, or the tile's V chunks before the scores, measured
// slower: 8-35% at 4K-26K keys)
struct R32 {
    uint4 u[4];
    float sc;
};
template <int KIND>
__device__ __forceinline__ void ld32(const uint8_t* cp, const uint8_t* sp, int c, R32& r) {
    if (KIND == kF16) {
#pragma unroll
        for (int i = 0; i < 4; ++i) r.u[i] = ((const uint4*) (cp + 64 * c))[i];
    } else if (KIND == kI8) {
        r.u[0] = ((const uint4*) (cp + 32 * c))[0];
        r.u[1] = ((const uint4*) (cp + 32 * c))[1];
        r.sc = __half2float(((const __half*) sp)[c]);
    } else if (KIND == kL4) {
        r.u[0] = *(const uint4*) (cp + 16 * c);
        r.sc = __half2float(((const __half*) sp)[c]);
    } else if (KIND == kT3) {
        const unsigned* src = (const unsigned*) (cp + 12 * c);
        r.u[0] = make_uint4(src[0], src[1], src[2], 0u);
        r.sc = *(const float*) sp;
    } else {
        r.u[0] = *(const uint4*) (cp + 16 * c);
        if (KIND == kT6) {
            const uint2 hi = *(const uint2*) (cp + kD / 2 + 8 * c);
            r.u[1] = make_uint4(hi.x, hi.y, 0u, 0u);
        }
        r.sc = *(const float*) sp;
    }
}
template <int KIND>
__device__ __forceinline__ void dec32(const R32& r, const float* lv, __half2* o) {
    const float sc = r.sc;
    if (KIND == kF16) {
#pragma unroll
        for (int i = 0; i < 4; ++i) ((uint4*) o)[i] = r.u[i];
    } else if (KIND == kI8) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int8_t* qb = (const int8_t*) &r.u[i];
#pragma unroll
            for (int j = 0; j < 8; ++j) o[8 * i + j] = __floats2half2_rn(qb[2 * j] * sc, qb[2 * j + 1] * sc);
        }
    } else if (KIND == kL4) {
        const unsigned w4[4] = {r.u[0].x, r.u[0].y, r.u[0].z, r.u[0].w};
#pragma unroll
        for (int k = 0; k < 4; ++k)
#pragma unroll
            for (int j = 0; j < 4; ++j)
                o[4 * k + j] = __floats2half2_rn((float) ((int) ((w4[k] >> (8 * j)) & 15u) - 8) * sc,
                                                 (float) ((int) ((w4[k] >> (8 * j + 4)) & 15u) - 8) * sc);
    } else if (KIND == kT3) {
        const unsigned w0 = r.u[0].x, w1 = r.u[0].y, w2 = r.u[0].z;
        const unsigned g4[4] = {w0 & 0xffffffu, (w0 >> 24) | ((w1 & 0xffffu) << 8), (w1 >> 16) | ((w2 & 0xffu) << 16),
                                w2 >> 8};
#pragma unroll
        for (int k = 0; k < 4; ++k)
#pragma unroll
            for (int j = 0; j < 4; ++j)
                o[4 * k + j] = __floats2half2_rn(lv[(g4[k] >> (6 * j)) & 7u] * sc, lv[(g4[k] >> (6 * j + 3)) & 7u] * sc);
    } else {   // kT4, kT6: 32 nibbles (and their high 2 bits)
        const unsigned lw[4] = {r.u[0].x, r.u[0].y, r.u[0].z, r.u[0].w};
        const unsigned hw[4] = {r.u[1].x & 0xffffu, r.u[1].x >> 16, r.u[1].y & 0xffffu, r.u[1].y >> 16};
#pragma unroll
        for (int k = 0; k < 4; ++k)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                unsigned c0 = (lw[k] >> (8 * j)) & 15u, c1 = (lw[k] >> (8 * j + 4)) & 15u;
                if (KIND == kT6) {
                    c0 |= ((hw[k] >> (4 * j)) & 3u) << 4;
                    c1 |= ((hw[k] >> (4 * j + 2)) & 3u) << 4;
                }
                o[4 * k + j] = __floats2half2_rn(lv[c0] * sc, lv[c1] * sc);
            }
    }
}

// the same 32 values through a table of code PAIRS (FP16 half2 of the two codebook values): one shared-memory load a
// pair and one half2 multiply by the scale, instead of two loads, two multiplies and a conversion - the decode is
// this kernel's limit (memory: every K/V format at ~370 G values/s).  T6: 6 + 6 bits (4096 entries), T4: 4 + 4 (256),
// T3: 3 + 3 (64).  (The product is rounded twice - the value to FP16, then the scaled one - instead of once.)
template <int KIND> __host__ __device__ constexpr int lut_n() { return KIND == kT6 ? 4096 : KIND == kT4 ? 256 : KIND == kT3 ? 64 : 1; }
template <int KIND> __device__ __forceinline__ bool lut_kind() { return KIND == kT6 || KIND == kT4 || KIND == kT3; }
template <int KIND>
__device__ __forceinline__ void dec32_lut(const R32& r, const __half2* __restrict__ lut, __half2* o) {
    const __half2 sc = __float2half2_rn(r.sc);
    if (KIND == kT3) {
        const unsigned w0 = r.u[0].x, w1 = r.u[0].y, w2 = r.u[0].z;
        const unsigned g4[4] = {w0 & 0xffffffu, (w0 >> 24) | ((w1 & 0xffffu) << 8), (w1 >> 16) | ((w2 & 0xffu) << 16),
                                w2 >> 8};
#pragma unroll
        for (int k = 0; k < 4; ++k)
#pragma unroll
            for (int j = 0; j < 4; ++j) o[4 * k + j] = __hmul2(lut[(g4[k] >> (6 * j)) & 63u], sc);
    } else {
        const unsigned lw[4] = {r.u[0].x, r.u[0].y, r.u[0].z, r.u[0].w};
        const unsigned hw[4] = {r.u[1].x & 0xffffu, r.u[1].x >> 16, r.u[1].y & 0xffffu, r.u[1].y >> 16};
#pragma unroll
        for (int k = 0; k < 4; ++k)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                unsigned c0 = (lw[k] >> (8 * j)) & 15u, c1 = (lw[k] >> (8 * j + 4)) & 15u;
                if (KIND == kT6) {
                    c0 |= ((hw[k] >> (4 * j)) & 3u) << 4;
                    c1 |= ((hw[k] >> (4 * j + 2)) & 3u) << 4;
                    o[4 * k + j] = __hmul2(lut[c0 | (c1 << 6)], sc);
                } else {
                    o[4 * k + j] = __hmul2(lut[c0 | (c1 << 4)], sc);
                }
            }
    }
}
template <int KIND>
__device__ __forceinline__ void build_lut(__half2* lut, const float* cb, int t, int nt) {
    constexpr int bits = KIND == kT6 ? 6 : KIND == kT4 ? 4 : 3, m = (1 << bits) - 1;
    for (int i = t; i < lut_n<KIND>(); i += nt) lut[i] = __floats2half2_rn(cb[i & m], cb[i >> bits]);
}

constexpr int kMT = 128, kMLdQ = kD + 8, kMLdB = 32 + 8, kMLdS = kMT + 4, kMLdP = kMT + 8;
template <int KK, int VK, bool QS, bool PS>
__global__ void __launch_bounds__(128) gqa_attn_mma_kernel(const float* __restrict__ q, GqaKv kv, int n_pos,
                                                           float scale, int chunk, float* __restrict__ part, int use_lut) {
    using namespace nvcuda;
    constexpr int G = 8;
    __shared__ __align__(32) __half sqh[(QS ? 2 : 1) * G * kMLdQ];   // Q, FP16, [head][dim] (QS: then q - its FP16)
    __shared__ __align__(32) __half sbuf[4][32 * kMLdB];        // a warp's 32 keys x 32 dims of K or V, FP16
    __shared__ __align__(32) float sS[G * kMLdS];               // the tile's scores [head][key]; (3) a tile's P . V
    __shared__ __align__(32) __half sP[(PS ? 2 : 1) * G * kMLdP];    // the probabilities, FP16 [head][key] (PS: + rest)
    __shared__ float scorr[G], smx[G], sls[G];
    __shared__ float scb[2][64];
    __shared__ __half2 klut[lut_n<KK>()], vlut[lut_n<VK>()];
    const int kvh = blockIdx.x, split = blockIdx.y, t = threadIdx.x, lane = t & 31, w = t >> 5;
    load_cbs(scb, kv.kk, kv.vk, t, 128);
    if (use_lut) {
        __syncthreads();
        if (lut_kind<KK>()) build_lut<KK>(klut, scb[0], t, 128);
        if (lut_kind<VK>()) build_lut<VK>(vlut, scb[1], t, 128);
    }
    for (int i = t; i < G * kD; i += 128) {
        const float x = q[(size_t) kvh * G * kD + i] * scale;
        const __half hx = __float2half(x);
        sqh[(i / kD) * kMLdQ + i % kD] = hx;
        if (QS) sqh[G * kMLdQ + (i / kD) * kMLdQ + i % kD] = __float2half(x - __half2float(hx));
    }
    __syncthreads();
    const Keys ks = keys_of(kv, n_pos);
    const bool cached = kv.sel != nullptr;
    const Rows K = k_rows(kv, scb[0], cached), V = v_rows(kv, scb[1], cached);
    const int k_lo = split * chunk, k_hi = min(k_lo + chunk, ks.n_idx);
    float m_r[2] = {-INFINITY, -INFINITY}, l_r[2] = {0.0f, 0.0f};
    float o[G];
#pragma unroll
    for (int g = 0; g < G; ++g) o[g] = 0.0f;
    __half* B = sbuf[w];
    // B's row lane from 16 half2 (zeros for a masked key: its row may be stale), then the warp's 2 MMAs of 16
    const auto put = [&](const __half2* h, bool ok) {
        __syncwarp();   // (the buffer's previous rows read)
#pragma unroll
        for (int j = 0; j < 4; ++j)
            ((uint4*) (B + lane * kMLdB))[j] = ok ? ((const uint4*) h)[j] : make_uint4(0u, 0u, 0u, 0u);
        __syncwarp();
    };
    for (int k0 = k_lo; k0 < k_hi; k0 += kMT) {
        {   // (1) keys k0 + 32 w + lane
            const int i = k0 + 32 * w + lane;
            const bool ok = i < k_hi && ks.ok(i);
            const uint8_t* bp = ok ? K.base(ks.pos(i)) : nullptr;
            wmma::fragment<wmma::accumulator, 8, 32, 16, float> acc;
            wmma::fill_fragment(acc, 0.0f);
#pragma unroll 1
            for (int c = 0; c < kD / 32; ++c) {
                __half2 h[16];
                if (ok) {
                    R32 r;
                    ld32<KK>(K.codes(bp, kvh), K.scales(bp, kvh), c, r);
                    if (lut_kind<KK>() && use_lut) dec32_lut<KK>(r, klut, h);
                    else dec32<KK>(r, K.lv, h);
                }
                put(h, ok);
#pragma unroll
                for (int kk = 0; kk < 2; ++kk) {
                    wmma::fragment<wmma::matrix_a, 8, 32, 16, __half, wmma::row_major> fa;
                    wmma::fragment<wmma::matrix_b, 8, 32, 16, __half, wmma::col_major> fb;
                    wmma::load_matrix_sync(fa, sqh + 32 * c + 16 * kk, kMLdQ);
                    wmma::load_matrix_sync(fb, B + 16 * kk, kMLdB);   // (key n's dims at n * kMLdB: column-major)
                    wmma::mma_sync(acc, fa, fb, acc);
                    if (QS) {
                        wmma::load_matrix_sync(fa, sqh + G * kMLdQ + 32 * c + 16 * kk, kMLdQ);
                        wmma::mma_sync(acc, fa, fb, acc);
                    }
                }
            }
            wmma::store_matrix_sync(sS + 32 * w, acc, kMLdS, wmma::mem_row_major);
            __syncwarp();
            if (!ok)
#pragma unroll
                for (int g = 0; g < G; ++g) sS[g * kMLdS + 32 * w + lane] = -INFINITY;
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < 2; ++j) {   // (2) heads w, w + 4
            const int g = w + 4 * j;
            float* sp = sS + g * kMLdS;
            float mt = -INFINITY;
#pragma unroll
            for (int x = lane; x < kMT; x += 32) mt = fmaxf(mt, sp[x]);
            mt = wmax(mt);
            const float mn = fmaxf(m_r[j], mt);
            const float corr = m_r[j] == -INFINITY ? 0.0f : __expf(m_r[j] - mn);
            float ls = 0.0f;
#pragma unroll
            for (int x = lane; x < kMT; x += 32) {
                const float v = sp[x];
                const float pv = v == -INFINITY ? 0.0f : __expf(v - mn);
                const __half hp = __float2half(pv);
                sP[g * kMLdP + x] = hp;
                if (PS) sP[G * kMLdP + g * kMLdP + x] = __float2half(pv - __half2float(hp));
                ls += pv;
            }
            ls = wsum(ls);
            l_r[j] = l_r[j] * corr + ls;
            m_r[j] = mn;
            if (lane == 0) scorr[g] = corr;
        }
        __syncthreads();
        {   // (3) dims 32 w .. 32 w + 31, 32 keys at a time
            wmma::fragment<wmma::accumulator, 8, 32, 16, float> acc;
            wmma::fill_fragment(acc, 0.0f);
#pragma unroll
            for (int kb = 0; kb < kMT / 32; ++kb) {
                const int i = k0 + 32 * kb + lane;
                const bool ok = i < k_hi && ks.ok(i);
                __half2 h[16];
                if (ok) {
                    const uint8_t* bp = V.base(ks.pos(i));
                    R32 r;
                    ld32<VK>(V.codes(bp, kvh), V.scales(bp, kvh), w, r);
                    if (lut_kind<VK>() && use_lut) dec32_lut<VK>(r, vlut, h);
                    else dec32<VK>(r, V.lv, h);
                }
                put(h, ok);
#pragma unroll
                for (int kk = 0; kk < 2; ++kk) {
                    wmma::fragment<wmma::matrix_a, 8, 32, 16, __half, wmma::row_major> fa;
                    wmma::fragment<wmma::matrix_b, 8, 32, 16, __half, wmma::row_major> fb;
                    wmma::load_matrix_sync(fa, sP + 32 * kb + 16 * kk, kMLdP);
                    wmma::load_matrix_sync(fb, B + 16 * kk * kMLdB, kMLdB);   // (key k's dims at k * kMLdB)
                    wmma::mma_sync(acc, fa, fb, acc);
                    if (PS) {
                        wmma::load_matrix_sync(fa, sP + G * kMLdP + 32 * kb + 16 * kk, kMLdP);
                        wmma::mma_sync(acc, fa, fb, acc);
                    }
                }
            }
            // (sS is free: every warp has read its scores in (2))
            wmma::store_matrix_sync(sS + 32 * w, acc, kMLdS, wmma::mem_row_major);
            __syncwarp();
#pragma unroll
            for (int g = 0; g < G; ++g) o[g] = o[g] * scorr[g] + sS[g * kMLdS + t];
        }
        __syncthreads();   // (sS, sP, scorr: the next tile's)
    }
#pragma unroll
    for (int j = 0; j < 2; ++j)
        if (lane == 0) {
            smx[w + 4 * j] = m_r[j];
            sls[w + 4 * j] = l_r[j];
        }
    __syncthreads();
#pragma unroll
    for (int g = 0; g < G; ++g) {
        float* po = part + ((size_t) (kvh * G + g) * gridDim.y + split) * (kD + 2);
        po[t] = o[g];
        if (t == 0) {
            po[kD] = smx[g];
            po[kD + 1] = sls[g];
        }
    }
}
#endif

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) std::fprintf(stderr, "glm_gqa %s: %s\n", what, cudaGetErrorString(e));
}

}  // namespace

namespace {
// a mode's K and V kinds
void mode_kinds(int mode, int& kk, int& vk) {
    static const int K[] = {kF16, kI8, kI8, kT4, kT6, kT4, kT6}, V[] = {kF16, kI8, kL4, kT4, kT4, kT3, kT3};
    if (mode < 0 || mode > kGqaTq6V3) mode = kGqaFp16;
    kk = K[mode];
    vk = V[mode];
}
const char* const kModeNames[] = {"fp16", "int8", "k8v4", "tq4", "tq6v4", "tq4v3", "tq6v3"};
}  // namespace

// a view's V row: its code and scale bytes
void mode_bytes(const GqaKv& kv, int& cb, int& sb) {
    cb = code_bytes(kv.vk);
    sb = scale_bytes(kv.vk);
}

const char* gqa_kv_mode_name(int mode) { return mode >= 0 && mode <= kGqaTq6V3 ? kModeNames[mode] : "?"; }

int gqa_kv_mode_parse(const char* s) {
    if (s == nullptr) return -1;
    for (int m = 0; m <= kGqaTq6V3; ++m)
        if (std::strcmp(s, kModeNames[m]) == 0) return m;
    return -1;
}

size_t gqa_kv_rec_bytes(int n_kv, int mode) {
    int kk = 0, vk = 0;
    mode_kinds(mode, kk, vk);
    return (size_t) n_kv * (row_bytes(kk) + row_bytes(vk));
}

GqaKv gqa_kv_view(void* a, void* b, int64_t p0, int64_t rec, int64_t layer_off, int n_kv, int64_t max_ctx, int mode) {
    GqaKv kv;
    kv.a = a;
    kv.b = b;
    kv.p0 = p0;
    kv.rec = rec;
    kv.k_off = (int) layer_off;
    mode_kinds(mode, kv.kk, kv.vk);
    kv.n_kv = n_kv;
    kv.max_ctx = (int) max_ctx;
    return kv;
}

size_t gqa_attn_scratch_floats(int n_head, int max_pos) {
    // (the wide kernel's splits: at most kWideSplits a K/V head - see gqa_attn)
    const int n_split = std::max((max_pos + kChunk - 1) / kChunk, 256);
    return (size_t) n_head * n_split * (kD + 2) + (size_t) n_head * kD;
}

void gqa_prep(float* q, const float* k, const float* v, int T, int qld, int kld, const float* q_norm,
              const float* k_norm, float eps, int p0, float rope_base, int n_head, const GqaKv& kv, cudaStream_t s) {
    if (T <= 0) return;
    const float theta_scale = powf(rope_base, -2.0f / (float) kD);
    gqa_prep_kernel<<<dim3(T, n_head + 2 * kv.n_kv), kD, 0, s>>>(q, k, v, qld, kld, q_norm, k_norm, eps, p0, theta_scale,
                                                               n_head, kv);
    check("gqa_prep");
}

bool gqa_attn(const float* q, const GqaKv& kv, int n_pos, int n_head, float* scratch, void* out_q8_1,
              cudaStream_t s) {
    if (n_head % kv.n_kv != 0 || n_head / kv.n_kv > kMaxGroup || n_pos < 1) return false;
    // (a streamed view: the splits of its largest selection - those past the step's own selection are empty)
    const int n_keys = kv.sel != nullptr ? kv.max_sel * kGqaPage : n_pos;
    // STRATA_GLM_GQA_DEC= (default: the tensor cores, 8 heads a K/V head) | wide | tc | warp
    static const int dec = [] {
        const char* v = getenv("STRATA_GLM_GQA_DEC");
        return v == nullptr ? 0 : std::strcmp(v, "tc") == 0 ? 1 : std::strcmp(v, "warp") == 0 ? 2
             : std::strcmp(v, "wide") == 0 ? 3 : 0;
    }();
#if !defined(STRATA_USE_HIP)
    if (dec == 0 && n_head / kv.n_kv == 8) {
        // the tensor cores (gqa_attn_mma_kernel): 128-key tiles, ~40 splits a K/V head (20 or 80: 7-27% slower at 26K)
        static const int target = getenv("DBG_SPLITS") ? std::atoi(getenv("DBG_SPLITS")) : 40;
        const int tiles = (n_keys + kMT - 1) / kMT, per = std::max(1, (tiles + target - 1) / target), chunk = per * kMT;
        const int n_split = (n_keys + chunk - 1) / chunk;
        float* part = scratch;
        float* out = scratch + (size_t) n_head * n_split * (kD + 2);
        const dim3 grid(kv.n_kv, n_split);
        const float sc = 1.0f / sqrtf((float) kD);
        // q as two FP16 parts (its FP16 and the rest: two MMAs a step): with q in FP16 alone the KL against FP16 K/V
        // rose 0.062 -> 0.075 (tq6v3), with the two 0.065 as the F32 kernel's (~2% slower).  The probabilities as two
        // as well changed nothing.  (DBG_MMA_SPLIT: 0 none, 1 q, 2 P, 3 both)
        static const int msplit = getenv("DBG_MMA_SPLIT") ? std::atoi(getenv("DBG_MMA_SPLIT")) : 1;
        static const int use_lut = getenv("STRATA_GLM_GQA_LUT") ? std::atoi(getenv("STRATA_GLM_GQA_LUT")) : 1;
#define MMA2(a, b, x, y) gqa_attn_mma_kernel<a, b, x, y><<<grid, 128, 0, s>>>(q, kv, n_pos, sc, chunk, part, use_lut)
#define MMA(a, b)                                                                                                    \
    do {                                                                                                             \
        if (msplit == 0) MMA2(a, b, false, false);                                                                    \
        else if (msplit == 1) MMA2(a, b, true, false);                                                               \
        else if (msplit == 2) MMA2(a, b, false, true);                                                               \
        else MMA2(a, b, true, true);                                                                                 \
    } while (0)
        if (kv.kk == kF16) MMA(kF16, kF16);
        else if (kv.kk == kI8) { if (kv.vk == kI8) MMA(kI8, kI8); else MMA(kI8, kL4); }
        else if (kv.kk == kT4) { if (kv.vk == kT4) MMA(kT4, kT4); else MMA(kT4, kT3); }
        else { if (kv.vk == kT4) MMA(kT6, kT4); else MMA(kT6, kT3); }
#undef MMA
#undef MMA2
        check("gqa_attn_mma");
        gqa_combine_kernel<<<n_head, kD, 0, s>>>(part, n_split, out, (Q8_1*) out_q8_1, v_rot(kv));
        check("gqa_combine");
        return true;
    }
#endif
    if ((dec == 0 || dec == 3) && n_head / kv.n_kv == 8) {
        // (HIP, or STRATA_GLM_GQA_DEC=wide) splits of whole 128-key tiles, ~40 a K/V head (~320 blocks: four an SM on 80): more splits, more blocks at once
        // but a larger combine
        static const int target = getenv("DBG_SPLITS") ? std::atoi(getenv("DBG_SPLITS")) : 40;
        // (64-key tiles below 2K keys: twice the blocks, each a tile's chain of loads half as long; 128 past that,
        // where the tiles' fixed costs count - DBG_TK=<32|64|128> to force one)
        static const int tk_env = getenv("DBG_TK") ? std::atoi(getenv("DBG_TK")) : 0;
        const int TK = tk_env > 0 ? tk_env : n_keys <= 2048 ? 64 : 128;
        const int tiles = (n_keys + TK - 1) / TK, per = std::max(1, (tiles + target - 1) / target), chunk = per * TK;
        const int n_split = (n_keys + chunk - 1) / chunk;
        float* part = scratch;
        float* out = scratch + (size_t) n_head * n_split * (kD + 2);
        // (the V rows staged, codes then scales, 16-byte rows; FP16 V reads its rows directly)
        int vk_cb = 0, vk_sb = 0;
        mode_bytes(kv, vk_cb, vk_sb);
        const int vstride = vk_cb >= 256 ? 0 : (vk_cb + std::max(vk_sb, 8) + 15) / 16 * 16;
        const dim3 grid(kv.n_kv, n_split);
        const float sc = 1.0f / sqrtf((float) kD);
        const size_t smem = (size_t) TK * vstride;
#define WIDE(tk)                                                                                                   \
        if (vstride > 0) gqa_attn_wide_kernel<8, tk, true><<<grid, kST, smem, s>>>(q, kv, n_pos, sc, chunk, vstride, part); \
        else gqa_attn_wide_kernel<8, tk, false><<<grid, kST, 0, s>>>(q, kv, n_pos, sc, chunk, 0, part);
        if (TK == 32) { WIDE(32) } else if (TK == 64) { WIDE(64) } else { WIDE(128) }
#undef WIDE
        check("gqa_attn_wide");
        gqa_combine_kernel<<<n_head, kD, 0, s>>>(part, n_split, out, (Q8_1*) out_q8_1, v_rot(kv));
        check("gqa_combine");
        return true;
    }
    const int n_split = (n_keys + kChunk - 1) / kChunk;
    float* part = scratch;
    float* out = scratch + (size_t) n_head * n_split * (kD + 2);
#if !defined(STRATA_USE_HIP)
    // the tensor cores (sm_70+) unless STRATA_GLM_GQA_DEC_TC=0 (the warp-reduction kernel below)
    static const bool tc = [] {
        const char* v = getenv("STRATA_GLM_GQA_DEC_TC");
        int dev = 0, major = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        return major >= 7 && !(v != nullptr && v[0] == '0') && dec != 2;
    }();
    // (below 128 positions the warp kernel is faster: 6 against 10 us a layer at 1 position, 21 against 31 at 37)
    if (tc && std::min(n_keys, n_pos) >= 128)
        gqa_attn_tc_kernel<<<dim3(kv.n_kv, n_split), kDW * 32, 0, s>>>(q, kv, n_pos, n_head, 1.0f / sqrtf((float) kD), part);
    else
#endif
    gqa_attn_kernel<<<dim3(kv.n_kv, n_split), 256, 0, s>>>(q, kv, n_pos, n_head, 1.0f / sqrtf((float) kD), part);
    check("gqa_attn");
    gqa_combine_kernel<<<n_head, kD, 0, s>>>(part, n_split, out, (Q8_1*) out_q8_1, v_rot(kv));
    check("gqa_combine");
    return true;
}

bool gqa_attn_rows(const float* q, int qld, const GqaKv& kv, int p0, int T, int n_head, void* out16, int old,
                   cudaStream_t s) {
    if (n_head % kv.n_kv != 0 || T <= 0) return false;
#if !defined(STRATA_USE_HIP)
    // the tensor cores (sm_70+) unless STRATA_GLM_GQA_TC=0 (the F32 kernel below, the reference)
    static const bool tc = [] {
        const char* v = getenv("STRATA_GLM_GQA_TC");
        int dev = 0, major = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        return major >= 7 && !(v != nullptr && v[0] == '0') &&
               cudaFuncSetAttribute(gqa_attn_rows_tc_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int) kTcSmem) == cudaSuccess;
    }();
    if (tc) {
        gqa_attn_rows_tc_kernel<<<dim3((T + kTQ - 1) / kTQ, n_head), kTW * 32, kTcSmem, s>>>(
            q, qld, kv, p0, T, n_head, 1.0f / sqrtf((float) kD), (__half*) out16, old);
        check("gqa_attn_rows_tc");
        if (v_rot(kv)) {
            gqa_unrotate_rows_kernel<<<dim3(T, n_head), kD, 0, s>>>((__half*) out16, n_head, old, v_rot(kv));
            check("gqa_unrotate_rows");
        }
        return true;
    }
#endif
    gqa_attn_rows_kernel<<<dim3((T + kQT - 1) / kQT, n_head), 256, 0, s>>>(q, qld, kv, p0, T, n_head,
                                                                          1.0f / sqrtf((float) kD), (__half*) out16, old);
    check("gqa_attn_rows");
    if (v_rot(kv)) {
        gqa_unrotate_rows_kernel<<<dim3(T, n_head), kD, 0, s>>>((__half*) out16, n_head, old, v_rot(kv));
        check("gqa_unrotate_rows");
    }
    return true;
}

void gqa_stream_summarize(const GqaKv& src, const GqaStream& st, int pos_lo, int pos_hi, cudaStream_t s) {
    if (pos_hi <= pos_lo) return;
    const int p0 = pos_lo / kGqaPage, p1 = (pos_hi + kGqaPage - 1) / kGqaPage;
    GqaKv v = src;
    v.ptab = nullptr;
    v.sel = nullptr;
    if (pos_hi - pos_lo == 1)   // (the decode: rounding the summary through FP16 once a position, as the full re-read did)
        gqa_summ_one_kernel<<<src.n_kv, kD, 0, s>>>(v, st, pos_lo);
    else
        gqa_summ_kernel<<<dim3(p1 - p0, src.n_kv), kD, 0, s>>>(v, st, p0, pos_hi);
    check("gqa_stream_summarize");
}

void gqa_stream_select(const float* q, int n_head, int n_kv, const GqaStream& st, int n_pos, int budget_pages,
                       int sink_pages, int recent_pages, const GqaSelBufs& b, cudaStream_t s) {
    const int np = (n_pos + kGqaPage - 1) / kGqaPage;
    const int total = std::min(np, sink_pages + recent_pages + budget_pages);
    static const bool old_sel = getenv("STRATA_GLM_KV_SEL1") != nullptr;   // (A/B: the one-block kernels)
    if (total >= np) {
        gqa_sel_all_kernel<<<1, 1024, 0, s>>>(np, b.sel, b.nsel);
    } else if (old_sel || b.hm == nullptr || b.sc == nullptr || n_head > 64 || n_head / n_kv > 16) {
        gqa_sel_ub_kernel<<<dim3(n_kv, (np + 127) / 128), 128, 0, s>>>(q, n_head, n_kv, st, np, b.ub);
        gqa_sel_pick_kernel<<<1, kSelT, (size_t) np * sizeof(float), s>>>(b.ub, n_head, np, budget_pages, sink_pages,
                                                                         recent_pages, b.sel, b.nsel, b.max_sel);
    } else {
        gqa_sel_init_kernel<<<1, 64, 0, s>>>(b.hm, b.nsel, n_head);
        gqa_sel_ub2_kernel<<<dim3(n_kv, (np + 31) / 32), 32 * (n_head / n_kv), 0, s>>>(q, n_head, n_kv, st, np, b.ub,
                                                                                         b.hm);
        gqa_sel_score_kernel<<<(np + 255) / 256, 256, 0, s>>>(b.ub, b.hm, n_head, np, sink_pages, recent_pages, b.sc);
        gqa_sel_pick2_kernel<<<(np + 31) / 32, 256, (size_t) np * sizeof(float), s>>>(b.sc, np, total, b.sel, b.nsel,
                                                                                     b.max_sel);
    }
    check("gqa_stream_select");
}

void gqa_stream_resolve(const GqaStream& st, int slice, const GqaSelBufs& b, cudaStream_t s) {
    const size_t tab = (size_t) 3 * st.R * sizeof(int);
    if (tab > 48 * 1024) {
        static bool opted = false;
        if (!opted) {
            cudaFuncSetAttribute(gqa_resolve_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) std::min<size_t>(tab, 96 * 1024));
            opted = true;
        }
    }
    gqa_resolve_kernel<<<1, 1024, tab, s>>>(st, b);
    gqa_page_copy_kernel<<<b.max_sel, 256, 0, s>>>(st, b, slice);
    check("gqa_stream_resolve");
}

void gqa_stream_invalidate(const GqaStream& st, int page_lo, int page_hi, cudaStream_t s) {
    if (page_hi <= page_lo) return;
    gqa_invalidate_kernel<<<(page_hi - page_lo + 255) / 256, 256, 0, s>>>(st, page_lo, page_hi);
    check("gqa_stream_invalidate");
}

void gqa_stream_reset(const GqaStream& st, cudaStream_t s) {
    cudaMemsetAsync(st.ptab, 0xff, (size_t) st.max_pages * sizeof(int), s);
    cudaMemsetAsync(st.spage, 0xff, (size_t) st.R * sizeof(int), s);
    cudaMemsetAsync(st.sepoch, 0, (size_t) st.R * sizeof(int), s);
    cudaMemsetAsync(st.sref, 0, (size_t) st.R * sizeof(int), s);
    cudaMemsetAsync(st.ctl, 0, 2 * sizeof(int), s);
}

}  // namespace strata::kernels::glmf
