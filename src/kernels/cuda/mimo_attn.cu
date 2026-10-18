// src/kernels/cuda/mimo_attn.cu - mimo2's attention (include/strata/kernels/mimo_attn.hpp): rope + the FP16 K / V
// records, a tensor-core prompt kernel (Hy3's gqa_attn_rows_tc_kernel widened to K 192 / V 128, with the window and
// the sink) and a split decode kernel (one block a K/V head and 128 keys: the query heads sharing the K/V head read each
// key once).
#include "strata/kernels/mimo_attn.hpp"

#include <cuda_fp16.h>
#if !defined(STRATA_USE_HIP)
#include <mma.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace strata::kernels::mimo {
namespace {

__device__ __forceinline__ int64_t slot_of(const Kv& kv, int64_t pos) {
    return kv.ring ? pos % kv.cap : pos - kv.pos0;
}
__device__ __forceinline__ const uint8_t* k_rec(const Kv& kv, int64_t pos) {
    return (const uint8_t*) kv.k + (size_t) slot_of(kv, pos) * rec_k_bytes(kv.n_kv, kv.q8);
}
__device__ __forceinline__ const uint8_t* v_rec(const Kv& kv, int64_t pos) {
    return (const uint8_t*) kv.v + (size_t) slot_of(kv, pos) * rec_v_bytes(kv.n_kv, kv.q8);
}
// 8 values of a K row (dims 8c ..) as 8 halves
__device__ __forceinline__ uint4 k8h(const Kv& kv, int64_t pos, int h, int c) {
    const uint8_t* r = k_rec(kv, pos);
    if (!kv.q8) return ((const uint4*) ((const __half*) r + (size_t) h * kDK))[c];
    const float sc = __half2float(((const __half*) (r + (size_t) kv.n_kv * kDK))[h * (kDK / 32) + c / 4]);
    const uint2 u = *(const uint2*) (r + (size_t) h * kDK + 8 * c);
    const int8_t* b = (const int8_t*) &u;
    uint4 o;
    __half2* oh = (__half2*) &o;
#pragma unroll
    for (int e = 0; e < 4; ++e) oh[e] = __floats2half2_rn(b[2 * e] * sc, b[2 * e + 1] * sc);
    return o;
}
// the same for a V row (dims 8c ..)
__device__ __forceinline__ uint4 v8h(const Kv& kv, int64_t pos, int h, int c) {
    const uint8_t* r = v_rec(kv, pos);
    if (!kv.q8) return ((const uint4*) ((const __half*) r + (size_t) h * kDV))[c];
    const float sc = __half2float(((const __half*) (r + (size_t) kv.n_kv * kDV))[h * (kDV / 32) + c / 4]);
    const uint2 u = *(const uint2*) (r + (size_t) h * kDV + 8 * c);
    const int8_t* b = (const int8_t*) &u;
    uint4 o;
    __half2* oh = (__half2*) &o;
#pragma unroll
    for (int e = 0; e < 4; ++e) oh[e] = __floats2half2_rn(b[2 * e] * sc, b[2 * e + 1] * sc);
    return o;
}
// 4 values of a V row (dims 4 lane ..) as floats
__device__ __forceinline__ void v4f(const Kv& kv, int64_t pos, int h, int lane, float f[4]) {
    const uint8_t* r = v_rec(kv, pos);
    if (!kv.q8) {
        const uint2 u = *(const uint2*) ((const __half*) r + (size_t) h * kDV + 4 * lane);
        const float2 a = __half22float2(*(const __half2*) &u.x), b = __half22float2(*(const __half2*) &u.y);
        f[0] = a.x;
        f[1] = a.y;
        f[2] = b.x;
        f[3] = b.y;
        return;
    }
    const float sc = __half2float(((const __half*) (r + (size_t) kv.n_kv * kDV))[h * (kDV / 32) + lane / 8]);
    const uint32_t u = *(const uint32_t*) (r + (size_t) h * kDV + 4 * lane);
    const int8_t* b = (const int8_t*) &u;
#pragma unroll
    for (int e = 0; e < 4; ++e) f[e] = b[e] * sc;
}
// a head's row of n floats into a slot's record: FP16, or int8 with an absmax / 127 scale (FP16) for each 32 values -
// a warp a head, a lane a value of each block
__device__ __forceinline__ void put_row(const float* x, int n, uint8_t* rec, int n_kv, int h, bool q8, int lane) {
    if (!q8) {
        __half* d = (__half*) rec + (size_t) h * n;
        for (int i = lane; i < n; i += 32) d[i] = __float2half(x[i]);
        return;
    }
    int8_t* d = (int8_t*) rec + (size_t) h * n;
    __half* scales = (__half*) (rec + (size_t) n_kv * n) + (size_t) h * (n / 32);
    for (int b = 0; b < n / 32; ++b) {
        const float v = x[b * 32 + lane];
        float mx = fabsf(v);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
        // (the scale as stored: the codes are rounded against the FP16 value the reader multiplies by)
        const __half sh = __float2half(mx > 0.0f ? mx / 127.0f : 1.0f);
        const float sc = __half2float(sh);
        d[b * 32 + lane] = (int8_t) __float2int_rn(fminf(127.0f, fmaxf(-127.0f, v / sc)));
        if (lane == 0) scales[b] = sh;
    }
}

// ---- rope (NeoX: pairs i, i + n_rot / 2 over the first n_rot dims, theta = pos * base^(-2 i / n_rot), as ggml's
// rope_neox) on q and k, then k and v into the records
__global__ void __launch_bounds__(256) prep_kernel(float* __restrict__ qkv, int64_t ld, int64_t p0, int n_head,
                                                   int n_rot, float theta_scale, Kv kv) {
    const int t = blockIdx.x;
    const int64_t pos = p0 + t;
    float* row = qkv + (size_t) t * ld;
    const int half_rot = n_rot / 2, heads = n_head + kv.n_kv;
    for (int i = threadIdx.x; i < heads * half_rot; i += blockDim.x) {
        const int h = i / half_rot, j = i % half_rot;
        float* x = row + (size_t) h * kDK;   // (q heads, then k heads: both 192 wide, k right after q)
        const float theta = (float) pos * powf(theta_scale, (float) j);
        float sn, cs;
        sincosf(theta, &sn, &cs);
        const float a = x[j], b = x[j + half_rot];
        x[j] = a * cs - b * sn;
        x[j + half_rot] = a * sn + b * cs;
    }
    __syncthreads();
    const float* k = row + (size_t) n_head * kDK;
    const float* v = k + (size_t) kv.n_kv * kDK;
    const int64_t sl = slot_of(kv, pos);
    uint8_t* kd = (uint8_t*) kv.k + (size_t) sl * rec_k_bytes(kv.n_kv, kv.q8);
    uint8_t* vd = (uint8_t*) kv.v + (size_t) sl * rec_v_bytes(kv.n_kv, kv.q8);
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = (int) (blockDim.x >> 5);
    for (int h = w; h < kv.n_kv; h += nw) {
        put_row(k + (size_t) h * kDK, kDK, kd, kv.n_kv, h, kv.q8, lane);
        put_row(v + (size_t) h * kDV, kDV, vd, kv.n_kv, h, kv.q8, lane);
    }
}

__global__ void __launch_bounds__(256) copy_kernel(Kv src, Kv dst, int64_t lo) {
    const int64_t pos = lo + blockIdx.x;   // (both views in one format)
    const size_t rk = rec_k_bytes(src.n_kv, src.q8), rv = rec_v_bytes(src.n_kv, src.q8);
    const uint2* ks = (const uint2*) ((const uint8_t*) src.k + (size_t) slot_of(src, pos) * rk);
    const uint2* vs = (const uint2*) ((const uint8_t*) src.v + (size_t) slot_of(src, pos) * rv);
    uint2* kd = (uint2*) ((uint8_t*) dst.k + (size_t) slot_of(dst, pos) * rk);
    uint2* vd = (uint2*) ((uint8_t*) dst.v + (size_t) slot_of(dst, pos) * rv);
    for (int i = threadIdx.x; i < (int) (rk / 8); i += blockDim.x) kd[i] = ks[i];
    for (int i = threadIdx.x; i < (int) (rv / 8); i += blockDim.x) vd[i] = vs[i];
}

// ---- decode / few rows: block (K/V head, a split of kpb keys, row), 4 warps taking 32-key groups in turn, one key a
// lane.  The G query heads of the K/V head score the lane's key from shared memory; each warp keeps an online softmax
// per head across its groups and sums the V rows with the lanes over the 128 dims.  The split's (m, l, O) per query
// head go to part, merged with the sink by combine_kernel.  The splits are sized so that a layer is about two blocks
// an SM (a 128-key split a block left a full-context layer at 19.5K positions with 612 blocks and a 153-part combine).
constexpr int kSplit = 128, kDecW = 4, kMaxG = 16, kPart = kDV + 2;
// a decode row's split layout: its keys' first position and the keys a split - the same for a row alone (T = 1) and
// in a verify window, so a window's row sums its keys in the order the token path does
__host__ __device__ __forceinline__ void dec_layout(int64_t pos, int window, int n_kv, int keys_env, int64_t& base,
                                                    int64_t& kpb, int& n_split) {
    base = window > 0 ? (pos - window + 1 > 0 ? pos - window + 1 : 0) : 0;
    const int64_t keys = pos + 1 - base;
    kpb = (keys * n_kv + 159) / 160;
    if (kpb < 128) kpb = 128;
    if (kpb < keys_env) kpb = keys_env;
    kpb = (kpb + 31) / 32 * 32;
    n_split = (int) ((keys + kpb - 1) / kpb);
}

__global__ void __launch_bounds__(kDecW * 32) dec_kernel(const float* __restrict__ q, int64_t qld, int64_t p0,
                                                         int n_head, Kv kv, int window, float scale, int keys_env,
                                                         int n_split, float* __restrict__ part) {
    const int kvh = blockIdx.x, sp = blockIdx.y, t = blockIdx.z;
    const int G = n_head / kv.n_kv;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int64_t pos = p0 + t, lo = window > 0 ? max((int64_t) 0, pos - window + 1) : 0;
    int64_t base, kpb;
    int n_mine;
    dec_layout(pos, window, kv.n_kv, keys_env, base, kpb, n_mine);
    __shared__ float sq[kMaxG][kDK];
    __shared__ float sm[kDecW][kMaxG], sl[kDecW][kMaxG];
    __shared__ float so[kDecW][kMaxG][kDV];
    for (int i = threadIdx.x; i < G * kDK; i += blockDim.x)
        sq[i / kDK][i % kDK] = q[(size_t) t * qld + (size_t) (kvh * G + i / kDK) * kDK + i % kDK] * scale;
    __syncthreads();
    // (a split past this row's own: empty - kb0 >= kb1)
    const int64_t kb0 = sp < n_mine ? base + (int64_t) sp * kpb : pos + 1, kb1 = min(kb0 + kpb, pos + 1);
    float m[kMaxG], l[kMaxG], acc[kMaxG][4];
#pragma unroll
    for (int g = 0; g < kMaxG; ++g) {
        m[g] = -INFINITY;
        l[g] = 0.0f;
        acc[g][0] = acc[g][1] = acc[g][2] = acc[g][3] = 0.0f;
    }
    for (int64_t g0 = kb0 + w * 32; g0 < kb1; g0 += kDecW * 32) {
        if (g0 + 31 < lo) continue;   // (the group is below the window: uniform across the warp)
        const int64_t kp = g0 + lane;
        const bool ok = kp >= lo && kp < kb1;
        float s[kMaxG];
#pragma unroll
        for (int g = 0; g < kMaxG; ++g) s[g] = 0.0f;
        if (ok) {
            for (int c = 0; c < kDK / 8; ++c) {
                const uint4 u = k8h(kv, kp, kvh, c);
                const __half2* hh = (const __half2*) &u;
                float kf[8];
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const float2 f = __half22float2(hh[e]);
                    kf[2 * e] = f.x;
                    kf[2 * e + 1] = f.y;
                }
#pragma unroll
                for (int g = 0; g < kMaxG; ++g) {
                    if (g >= G) break;
                    float a = 0.0f;
#pragma unroll
                    for (int e = 0; e < 8; ++e) a += sq[g][c * 8 + e] * kf[e];
                    s[g] += a;
                }
            }
        }
        float p[kMaxG];
#pragma unroll
        for (int g = 0; g < kMaxG; ++g) {
            if (g >= G) break;
            float mx = ok ? s[g] : -INFINITY;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            const float mn = fmaxf(m[g], mx);   // (mx is finite: the group has a key in range)
            const float corr = m[g] == -INFINITY ? 0.0f : __expf(m[g] - mn);
            p[g] = ok ? __expf(s[g] - mn) : 0.0f;
            float sum = p[g];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
            l[g] = l[g] * corr + sum;
            m[g] = mn;
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[g][e] *= corr;
        }
        for (int j = 0; j < 32; ++j) {
            const int64_t kj = g0 + j;
            if (kj < lo || kj >= kb1) continue;   // (uniform across the warp)
            float vf[4];
            v4f(kv, kj, kvh, lane, vf);
#pragma unroll
            for (int g = 0; g < kMaxG; ++g) {
                if (g >= G) break;
                const float pj = __shfl_sync(0xffffffffu, p[g], j);
                acc[g][0] += pj * vf[0];
                acc[g][1] += pj * vf[1];
                acc[g][2] += pj * vf[2];
                acc[g][3] += pj * vf[3];
            }
        }
    }
#pragma unroll
    for (int g = 0; g < kMaxG; ++g) {
        if (g >= G) break;
        if (lane == 0) {
            sm[w][g] = m[g];
            sl[w][g] = l[g];
        }
#pragma unroll
        for (int e = 0; e < 4; ++e) so[w][g][4 * lane + e] = acc[g][e];
    }
    __syncthreads();
    // merge the 4 warps into the split's part: thread d of the 128 over the dims, every head
    const int d = threadIdx.x;
    for (int g = 0; g < G; ++g) {
        float M = -INFINITY;
        for (int i = 0; i < kDecW; ++i) M = fmaxf(M, sm[i][g]);
        float L = 0.0f, O = 0.0f;
        if (M != -INFINITY)
            for (int i = 0; i < kDecW; ++i) {
                const float c = sm[i][g] == -INFINITY ? 0.0f : __expf(sm[i][g] - M);
                L += sl[i][g] * c;
                O += so[i][g][d] * c;
            }
        float* pt = part + (((size_t) t * n_head + kvh * G + g) * n_split + sp) * kPart;
        pt[2 + d] = O;
        if (d == 0) {
            pt[0] = M;
            pt[1] = L;
        }
    }
}

__global__ void __launch_bounds__(kDV) combine_kernel(const float* __restrict__ part, int n_split, int n_head,
                                                      const float* __restrict__ sinks, float out_scale,
                                                      float* __restrict__ out, int64_t ldo) {
    const int h = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    const float* pt = part + ((size_t) t * n_head + h) * n_split * kPart;
    const float sink = sinks != nullptr ? sinks[h] : -INFINITY;
    float M = sink;
    for (int i = 0; i < n_split; ++i) M = fmaxf(M, pt[(size_t) i * kPart]);
    float L = sinks != nullptr ? __expf(sink - M) : 0.0f, O = 0.0f;
    for (int i = 0; i < n_split; ++i) {
        const float mi = pt[(size_t) i * kPart];
        if (mi == -INFINITY) continue;
        const float c = __expf(mi - M);
        L += pt[(size_t) i * kPart + 1] * c;
        O += pt[(size_t) i * kPart + 2 + d] * c;
    }
    out[(size_t) t * ldo + (size_t) h * kDV + d] = L > 0.0f ? O / L * out_scale : 0.0f;
}

#if !defined(STRATA_USE_HIP)
// ---- the prompt on the tensor cores: Hy3's one-pass kernel (gqa_attn_rows_tc_kernel) with K 192 / V 128.  Block (128
// query rows, one query head), 8 warps of 16 rows with Q in 12 fragments; TK-key tiles of K and V through shared memory.
// The sink starts each row's softmax (m = sink, l = 1: its logit with no value); the window bounds the keys below.
// <64, false>: 64-key tiles, P beside S (96 KB: one block an SM); <32, true>: 32-key tiles with P written over the
// warp's S area once its scores are in registers, Q staged in two halves (39 KB: two blocks an SM).
constexpr int kTQ = 128, kTW = 8;
constexpr int kLdK = kDK + 8, kLdV = kDV + 8;
template <int TK, bool PS>
struct TcCfg {
    static constexpr int KH = TK / 2, LdS = TK + 4, LdP = TK + 8;
    static constexpr size_t warpS = (size_t) 16 * LdS * 4, warpP = PS ? 0 : (size_t) 16 * LdP * 2;
    static constexpr size_t tile = (size_t) TK * kLdK * 2 + (size_t) TK * kLdV * 2 + kTW * (warpS + warpP);
    static constexpr size_t qstage = (size_t) (PS ? kTQ / 2 : kTQ) * kLdK * 2;
    static constexpr size_t smem = tile > qstage ? tile : qstage;
    static_assert(!PS || (size_t) 16 * LdP * 2 <= warpS, "P must fit the warp's S area");
};
template <int TK, bool PS>
__global__ void __launch_bounds__(kTW * 32) rows_tc_kernel(const float* __restrict__ q, int64_t qld, Kv kv, int64_t p0,
                                                          int T, int n_head, int window, const float* __restrict__ sinks,
                                                          float scale, float out_scale, float* __restrict__ out,
                                                          int64_t ldo) {
    using namespace nvcuda;
    using C = TcCfg<TK, PS>;
    constexpr int KH = C::KH, LdS = C::LdS, LdP = C::LdP;
    extern __shared__ __align__(32) unsigned char smem[];
    __half* sK = (__half*) smem;                       // [TK][kLdK]
    __half* sV = sK + TK * kLdK;                       // [TK][kLdV]
    unsigned char* sW = (unsigned char*) (sV + TK * kLdV);   // per warp: S (and, !PS, P after it)
    const int h = blockIdx.y, r0 = blockIdx.x * kTQ, t = threadIdx.x, lane = t & 31, w = t >> 5;
    const int kvh = h / (n_head / kv.n_kv);
    // Q (scaled) -> FP16 staging -> each warp's 12 fragments (PS: in two halves of 64 rows - warps 0-3, then 4-7)
    __half* sQ = (__half*) smem;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fq[kDK / 16];
    for (int half = 0; half < (PS ? 2 : 1); ++half) {
        const int rows = PS ? kTQ / 2 : kTQ, rb = half * rows;
        if (half > 0) __syncthreads();   // (the first half's fragments loaded)
        for (int i = t; i < rows * kDK; i += kTW * 32) {
            const int r = i / kDK, d = i % kDK, row = r0 + rb + r;
            sQ[r * kLdK + d] = __float2half(row < T ? q[(size_t) row * qld + (size_t) h * kDK + d] * scale : 0.0f);
        }
        __syncthreads();
        if (w * 16 >= rb && w * 16 < rb + rows) {
#pragma unroll
            for (int k = 0; k < kDK / 16; ++k) wmma::load_matrix_sync(fq[k], sQ + (w * 16 - rb) * kLdK + k * 16, kLdK);
        }
    }
    float* Sw = (float*) (sW + (size_t) w * (C::warpS + C::warpP));
    __half* Pw = PS ? (__half*) Sw : (__half*) ((unsigned char*) Sw + C::warpS);
    const int ri = lane >> 1, half_i = lane & 1;
    const int row = r0 + w * 16 + ri;
    const int64_t pos = p0 + row;
    const bool row_ok = row < T;
    const int64_t last = p0 + min(r0 + kTQ, T) - 1;
    const int64_t wlast = p0 + min(r0 + w * 16 + 16, T) - 1;
    const int64_t lo_blk = window > 0 ? max((int64_t) 0, p0 + r0 - window + 1) : 0;
    const int64_t wfirst = p0 + r0 + w * 16;
    const int64_t lo_row = window > 0 ? max((int64_t) 0, pos - window + 1) : 0;
    const auto load_tile = [&](int64_t k0) {
        for (int i = t; i < TK * (kDK / 8 + kDV / 8); i += kTW * 32) {
            const int r = i / (kDK / 8 + kDV / 8), c = i % (kDK / 8 + kDV / 8);
            const int64_t kp = k0 + r;
            const bool ok = kp >= lo_blk && kp <= last;
            uint4 u = make_uint4(0u, 0u, 0u, 0u);
            if (c < kDK / 8) {
                if (ok) u = k8h(kv, kp, kvh, c);
                *(uint4*) (sK + r * kLdK + c * 8) = u;
            } else {
                const int cv = c - kDK / 8;
                if (ok) u = v8h(kv, kp, kvh, cv);
                *(uint4*) (sV + r * kLdV + cv * 8) = u;
            }
        }
    };
    constexpr float kLazy = 8.0f;
    float m = sinks != nullptr ? sinks[h] : -INFINITY, l = sinks != nullptr ? 1.0f : 0.0f;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fo[kDV / 16];
#pragma unroll
    for (int n = 0; n < kDV / 16; ++n) wmma::fill_fragment(fo[n], 0.0f);
    for (int64_t k0 = lo_blk / TK * TK; k0 <= last; k0 += TK) {
        __syncthreads();
        load_tile(k0);
        __syncthreads();
        // (the warp's rows see none of these keys: above its last, or below its first row's window)
        if (k0 > wlast || (window > 0 && k0 + TK - 1 < wfirst - window + 1)) continue;
#pragma unroll
        for (int n = 0; n < TK / 16; ++n) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::fill_fragment(acc, 0.0f);
#pragma unroll
            for (int k = 0; k < kDK / 16; ++k) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
                wmma::load_matrix_sync(fb, sK + (n * 16) * kLdK + k * 16, kLdK);
                wmma::mma_sync(acc, fq[k], fb, acc);
            }
            wmma::store_matrix_sync(Sw + n * 16, acc, LdS, wmma::mem_row_major);
        }
        __syncwarp();
        float sv[KH], mt = -INFINITY;
#pragma unroll
        for (int j = 0; j < KH; ++j) {
            const int c = half_i * KH + j;
            const int64_t kp = k0 + c;
            sv[j] = row_ok && kp <= pos && kp >= lo_row ? Sw[ri * LdS + c] : -INFINITY;
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
        __syncwarp();   // (PS: every lane's scores read before P overwrites them)
        float ps = 0.0f;
#pragma unroll
        for (int j = 0; j < KH; ++j) {
            const float pj = sv[j] == -INFINITY ? 0.0f : __expf(sv[j] - m);
            ps += pj;
            Pw[ri * LdP + half_i * KH + j] = __float2half(pj);
        }
        l += ps + __shfl_xor_sync(0xffffffffu, ps, 1);
        __syncwarp();
        if (__any_sync(0xffffffffu, need)) {
            // (the output's rescale goes through a 16 x 16 float area: PS - after P in the same S area, else S)
            float* Rw = PS ? (float*) ((unsigned char*) Sw + (size_t) 16 * LdP * 2) : Sw;
            constexpr int LdR = PS ? 16 : LdS;
            static_assert(!PS || (size_t) 16 * LdP * 2 + 16 * 16 * 4 <= C::warpS, "the rescale area must fit too");
#pragma unroll
            for (int n = 0; n < kDV / 16; ++n) {
                wmma::store_matrix_sync(Rw, fo[n], LdR, wmma::mem_row_major);
                __syncwarp();
#pragma unroll
                for (int j = 0; j < 8; ++j) Rw[ri * LdR + half_i * 8 + j] *= corr;
                __syncwarp();
                wmma::load_matrix_sync(fo[n], Rw, LdR, wmma::mem_row_major);
                __syncwarp();
            }
        }
#pragma unroll
        for (int k = 0; k < TK / 16; ++k) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fp;
            wmma::load_matrix_sync(fp, Pw + k * 16, LdP);
#pragma unroll
            for (int n = 0; n < kDV / 16; ++n) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fv;
                wmma::load_matrix_sync(fv, sV + (k * 16) * kLdV + n * 16, kLdV);
                wmma::mma_sync(fo[n], fp, fv, fo[n]);
            }
        }
        __syncwarp();
    }
    const float inv_l = l > 0.0f ? out_scale / l : 0.0f;
    float* Ow = Sw;   // (16 x 16 floats at a time, LdS apart - inside the warp's S area)
#pragma unroll
    for (int n = 0; n < kDV / 16; ++n) {
        wmma::store_matrix_sync(Ow, fo[n], LdS, wmma::mem_row_major);
        __syncwarp();
        if (row_ok) {
            float* o = out + (size_t) row * ldo + (size_t) h * kDV + n * 16 + half_i * 8;
#pragma unroll
            for (int j = 0; j < 8; ++j) o[j] = Ow[ri * LdS + half_i * 8 + j] * inv_l;
        }
        __syncwarp();
    }
}
#endif

constexpr int kDecRows = 16;   // up to this many rows go through the decode kernel

}  // namespace

void prep(float* qkv, int64_t ld, int T, int64_t p0, int n_head, int n_rot, float base, const Kv& kv, cudaStream_t s) {
    if (T <= 0) return;
    prep_kernel<<<T, 256, 0, s>>>(qkv, ld, p0, n_head, n_rot, powf(base, -2.0f / (float) n_rot), kv);
}

size_t attn_scratch_floats(int n_head, int T, int64_t max_keys, int min_kv) {
#if defined(STRATA_USE_HIP)
    T = std::min(T, kDecRows);   // (no tensor-core prompt kernel: the rows go through the decode kernel 16 at a time)
#else
    if (T > kDecRows) return 0;
#endif
    // (attn's splits: at least kSplit keys each, about 160 blocks a layer - at most 160 / n_kv + 1 of them)
    const int64_t n_split = std::min<int64_t>((max_keys + T + kSplit) / kSplit + 1, 160 / std::max(1, min_kv) + 2);
    return (size_t) T * n_head * n_split * kPart;
}

void attn(const float* q, int64_t qld, int T, int64_t p0, int n_head, const Kv& kv, int window, const float* sinks,
          float scale, float out_scale, float* out, int64_t ldo, float* scratch, cudaStream_t s) {
    if (T <= 0) return;
#if defined(STRATA_USE_HIP)
    for (int r = 0; r + kDecRows < T; r += kDecRows)
        attn(q + (size_t) r * qld, qld, kDecRows, p0 + r, n_head, kv, window, sinks, scale, out_scale,
             out + (size_t) r * ldo, ldo, scratch, s);
    if (T > kDecRows) {
        const int r = (T - 1) / kDecRows * kDecRows;
        q += (size_t) r * qld;
        out += (size_t) r * ldo;
        p0 += r;
        T -= r;
    }
#endif
    if (T <= kDecRows && scratch != nullptr) {   // (no scratch: the prompt kernel, whatever T)
        // splits: about 160 blocks a layer (two an SM on a V100), at least 128 keys each, each row's own layout
        // (dec_layout); STRATA_MIMO_DEC_KEYS can only make them larger (the scratch holds at most ~160 / n_kv)
        static const int keys_env = getenv("STRATA_MIMO_DEC_KEYS") != nullptr ? std::atoi(getenv("STRATA_MIMO_DEC_KEYS")) : 0;
        int n_split = 1;
        for (int t = 0; t < T; ++t) {
            int64_t b, k;
            int n;
            dec_layout(p0 + t, window, kv.n_kv, keys_env, b, k, n);
            n_split = std::max(n_split, n);
        }
        dec_kernel<<<dim3(kv.n_kv, n_split, T), kDecW * 32, 0, s>>>(q, qld, p0, n_head, kv, window, scale, keys_env,
                                                                   n_split, scratch);
        combine_kernel<<<dim3(n_head, T), kDV, 0, s>>>(scratch, n_split, n_head, sinks, out_scale, out, ldo);
        return;
    }
#if !defined(STRATA_USE_HIP)
    // (the 64-key tiles: 32-key tiles with P over S - two blocks an SM - were slower, 28K prompt 6.48 vs 6.32 s)
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute(rows_tc_kernel<64, false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int) TcCfg<64, false>::smem);
        attr = true;
    }
    rows_tc_kernel<64, false><<<dim3((T + kTQ - 1) / kTQ, n_head), kTW * 32, TcCfg<64, false>::smem, s>>>(
        q, qld, kv, p0, T, n_head, window, sinks, scale, out_scale, out, ldo);
#endif
}

void copy(const Kv& src, const Kv& dst, int64_t lo, int64_t hi, cudaStream_t s) {
    if (hi <= lo) return;
    copy_kernel<<<(unsigned) (hi - lo), 256, 0, s>>>(src, dst, lo);
}

}  // namespace strata::kernels::mimo
