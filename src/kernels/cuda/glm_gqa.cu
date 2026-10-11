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
__host__ __device__ __forceinline__ int tq_code_bytes(int kind) { return row_bytes(kind) - 4; }
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

// a K or V row (position, head): its record (region a below p0, else b), the layer's K / V part, the head's row
struct Rows {
    const uint8_t* a;
    const uint8_t* b;
    long long p0, rec;
    int off, rb, kind;   // kind: kF16 .. kT6
    const float* cb;     // a TurboQuant kind's levels (shared memory)
    const int* ptab;     // KV streaming: the rows of the cached pages (page table -> slots, rec bytes a position)
    const uint8_t* slots;
    __device__ __forceinline__ const uint8_t* row(long long pos, int h) const {
        if (ptab != nullptr)
            return slots + ((size_t) ptab[pos / kGqaPage] * kGqaPage + (size_t) (pos % kGqaPage)) * rec + off + h * rb;
        return (pos < p0 ? a + pos * rec : b - (pos - p0) * rec) + off + h * rb;   // (region b grows down)
    }
    __device__ __forceinline__ void load4(long long pos, int h, int d, float& x0, float& x1, float& x2, float& x3) const {
        const uint8_t* rp = row(pos, h);
        if (kind >= kT3) {
            const float sc = *(const float*) (rp + tq_code_bytes(kind));
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
            x0 = cb[c0] * sc; x1 = cb[c1] * sc; x2 = cb[c2] * sc; x3 = cb[c3] * sc;
        } else if (kind == kL4) {
            const uint16_t qq = *(const uint16_t*) (rp + d / 2);
            const float sc = __half2float(((const __half*) (rp + kD / 2))[d / 32]);
            x0 = (float) ((int) (qq & 15) - 8) * sc;
            x1 = (float) ((int) ((qq >> 4) & 15) - 8) * sc;
            x2 = (float) ((int) ((qq >> 8) & 15) - 8) * sc;
            x3 = (float) ((int) ((qq >> 12) & 15) - 8) * sc;
        } else if (kind == kI8) {
            const char4 q = *(const char4*) (rp + d);
            const float sc = __half2float(((const __half*) (rp + kD))[d / 32]);
            x0 = q.x * sc; x1 = q.y * sc; x2 = q.z * sc; x3 = q.w * sc;
        } else {
            const __half2* hp = (const __half2*) ((const __half*) rp + d);
            const float2 u = __half22float2(hp[0]), w = __half22float2(hp[1]);
            x0 = u.x; x1 = u.y; x2 = w.x; x3 = w.y;
        }
    }
};
// (cached: the rows of the cached pages - a streamed view's selection; else the view's own records)
__device__ __forceinline__ Rows k_rows(const GqaKv& kv, const float* cb = nullptr, bool cached = false) {
    return {(const uint8_t*) kv.a, (const uint8_t*) kv.b, kv.p0, kv.rec, kv.k_off, row_bytes(kv.kk), kv.kk, cb,
            cached ? kv.ptab : nullptr, kv.slots};
}
__device__ __forceinline__ Rows v_rows(const GqaKv& kv, const float* cb = nullptr, bool cached = false) {
    return {(const uint8_t*) kv.a, (const uint8_t*) kv.b, kv.p0, kv.rec, kv.k_off + kv.n_kv * row_bytes(kv.kk),
            row_bytes(kv.vk), kv.vk, cb, cached ? kv.ptab : nullptr, kv.slots};
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
    uint8_t* rp = (uint8_t*) R.row(pos, h);
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
        if (t == 0) *(float*) (rp + tq_code_bytes(R.kind)) = den > 0.0f ? num / den : 0.0f;
    } else if (R.kind == kL4) {
        const float amax = wmax(fabsf(val));
        const float sc = amax / 7.0f;
        const int q = sc > 0.0f ? max(-7, min(7, __float2int_rn(val / sc))) : 0;
        const int qn = __shfl_xor_sync(0xffffffffu, q, 1);
        if ((t & 1) == 0) rp[t / 2] = (uint8_t) ((q + 8) | ((qn + 8) << 4));
        if ((t & 31) == 0) ((__half*) (rp + kD / 2))[t / 32] = __float2half(sc);
    } else if (R.kind == kI8) {
        const float amax = wmax(fabsf(val));
        const float sc = amax / 127.0f;
        const int q = sc > 0.0f ? __float2int_rn(val / sc) : 0;
        rp[t] = (uint8_t) (int8_t) max(-127, min(127, q));
        if ((t & 31) == 0) ((__half*) (rp + kD))[t / 32] = __float2half(sc);
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

__global__ void __launch_bounds__(kD) gqa_combine_kernel(const float* __restrict__ part, int n_split,
                                                         float* __restrict__ out, int vrot) {
    __shared__ float so[kD];
    const int h = blockIdx.x, t = threadIdx.x;
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
    if (vrot) {   // the output was made of rotated V rows - rotate it back (H, or D H)
        so[t] = acc / den;
        fwht128(so, t);
        out[(size_t) h * kD + t] = vrot == 2 ? so[t] * rsign(t) : so[t];
        return;
    }
    out[(size_t) h * kD + t] = acc / den;
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
    // the threshold: the largest x with count(score >= x) >= total (scores of unforced pages are <= 0)
    __shared__ float red[kSelT / 32];
    lo = wmax(-lo);
    if (lane == 0) red[w] = lo;
    __syncthreads();
    if (t < 32) {
        float v = t < kSelT / 32 ? red[t] : 0.0f;
        v = wmax(v);
        if (t == 0) red[0] = v;
    }
    __syncthreads();
    float a = -red[0] - 1.0f, b = 0.0f;   // count(>= a) = np >= total; count(>= b) may be < total
    for (int it = 0; it < 40; ++it) {
        const float mid = 0.5f * (a + b);
        int c = 0;
        for (int p = t; p < np; p += kSelT) c += sc[p] >= mid;
        if (t == 0) cnt = 0;
        __syncthreads();
        atomicAdd(&cnt, c);
        __syncthreads();
        if (cnt >= total) a = mid; else b = mid;
        __syncthreads();
    }
    if (t == 0) cnt = 0;
    __syncthreads();
    for (int p = t; p < np; p += kSelT)
        if (sc[p] >= a) {
            const int k = atomicAdd(&cnt, 1);
            if (k < max_sel) sel[k] = p;
        }
    __syncthreads();
    if (t == 0) *nsel = min(cnt, max_sel);
}

// one block of 1024: the selected pages already cached are marked used by this step (and referenced); the others are
// claimed CLOCK's victims (never a slot this step uses) -> miss / mslot / nmiss for the copy below
__global__ void __launch_bounds__(1024) gqa_resolve_kernel(GqaStream st, GqaSelBufs b) {
    __shared__ int s_ep, s_nm;
    const int t = threadIdx.x;
    if (t == 0) {
        s_ep = ++st.ctl[1];
        s_nm = 0;
    }
    __syncthreads();
    const int ep = s_ep, ns = *b.nsel;
    for (int i = t; i < ns; i += 1024) {
        const int p = b.sel[i], sl = st.ptab[p];
        if (sl >= 0) {
            st.sepoch[sl] = ep;
            st.sref[sl] = 1;
        } else {
            b.miss[atomicAdd(&s_nm, 1)] = p;
        }
    }
    __syncthreads();
    if (t != 0) return;
    int hand = st.ctl[0];
    for (int k = 0; k < s_nm; ++k) {
        int sl = -1;
        for (int step = 0; step < 4 * st.R; ++step) {
            const int c = hand;
            hand = hand + 1 == st.R ? 0 : hand + 1;
            if (st.sepoch[c] == ep) continue;
            if (st.sref[c]) {
                st.sref[c] = 0;
                continue;
            }
            sl = c;
            break;
        }
        if (sl < 0) {   // (cannot happen: R exceeds the largest selection)
            *b.nmiss = k;
            st.ctl[0] = hand;
            return;
        }
        const int old = st.spage[sl];
        if (old >= 0) st.ptab[old] = -1;
        const int p = b.miss[k];
        st.spage[sl] = p;
        st.ptab[p] = sl;
        st.sepoch[sl] = ep;
        st.sref[sl] = 1;
        b.mslot[k] = sl;
    }
    st.ctl[0] = hand;
    *b.nmiss = s_nm;
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
    const int n_split = (max_pos + kChunk - 1) / kChunk;
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
        return major >= 7 && !(v != nullptr && v[0] == '0');
    }();
    // (below 128 positions the warp kernel is faster: 6 against 10 us a layer at 1 position, 21 against 31 at 37)
    if (tc && std::min(n_keys, n_pos) >= 128)
        gqa_attn_tc_kernel<<<dim3(kv.n_kv, n_split), kDW * 32, 0, s>>>(q, kv, n_pos, n_head, 1.0f / sqrtf((float) kD), part);
    else
#endif
    gqa_attn_kernel<<<dim3(kv.n_kv, n_split), 256, 0, s>>>(q, kv, n_pos, n_head, 1.0f / sqrtf((float) kD), part);
    check("gqa_attn");
    gqa_combine_kernel<<<n_head, kD, 0, s>>>(part, n_split, out, v_rot(kv));
    check("gqa_combine");
    quantize_q8_1(out, out_q8_1, n_head * kD, s);
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
    gqa_summ_kernel<<<dim3(p1 - p0, src.n_kv), kD, 0, s>>>(v, st, p0, pos_hi);
    check("gqa_stream_summarize");
}

void gqa_stream_select(const float* q, int n_head, int n_kv, const GqaStream& st, int n_pos, int budget_pages,
                       int sink_pages, int recent_pages, const GqaSelBufs& b, cudaStream_t s) {
    const int np = (n_pos + kGqaPage - 1) / kGqaPage;
    if (np > sink_pages + recent_pages + budget_pages)
        gqa_sel_ub_kernel<<<dim3(n_kv, (np + 127) / 128), 128, 0, s>>>(q, n_head, n_kv, st, np, b.ub);
    gqa_sel_pick_kernel<<<1, kSelT, (size_t) np * sizeof(float), s>>>(b.ub, n_head, np, budget_pages, sink_pages,
                                                                     recent_pages, b.sel, b.nsel, b.max_sel);
    check("gqa_stream_select");
}

void gqa_stream_resolve(const GqaStream& st, int slice, const GqaSelBufs& b, cudaStream_t s) {
    gqa_resolve_kernel<<<1, 1024, 0, s>>>(st, b);
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
