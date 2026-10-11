// src/kernels/cuda/glm_gqa.cu - hy_v3's grouped-query attention (glm_fast.hpp: GqaKv, gqa_prep, gqa_attn,
// gqa_attn_rows).  The arithmetic is llama.cpp's hy-v3 graph (src/models/hy-v3.cpp): per-head RMS norms on q and k
// (llm_build_norm over head_dim), NeoX rope over the whole head (ggml-cuda/rope.cu's rope_neox: theta = pos *
// base^(-2i/n_dims), the pair (i, i + n_dims/2)), K and V in a cache, softmax(q.k / sqrt(head_dim)) . v.
//
// The cache (GqaKv): per layer K then V, rows [position][n_kv][128] - FP16, or INT8 codes with one FP16 scale per 32
// values (symmetric, absmax / 127; the codes of every row first, then the scales: 136 bytes a row against 256).
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

namespace strata::kernels::glmf {
namespace {

constexpr int kD = 128;           // head_dim (hy3_geometry requires it)
constexpr int kChunk = 256;       // positions per split of the decode attention
constexpr int kMaxGroup = 16;     // query heads per K/V head

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

// row r = pos * n_kv + head of one of the two caches
struct Rows {
    uint8_t* base;
    int q8;
    size_t n_rows;   // max_ctx * n_kv (where an INT8 cache's scales start)
    __device__ __forceinline__ void load4(size_t r, int d, float& a, float& b, float& c, float& e) const {
        if (q8) {
            const char4 q = *(const char4*) (base + r * kD + d);
            const float sc = __half2float(((const __half*) (base + n_rows * kD))[r * (kD / 32) + d / 32]);
            a = q.x * sc; b = q.y * sc; c = q.z * sc; e = q.w * sc;
        } else {
            const __half2* h = (const __half2*) ((const __half*) base + r * kD + d);
            const float2 x = __half22float2(h[0]), y = __half22float2(h[1]);
            a = x.x; b = x.y; c = y.x; e = y.y;
        }
    }
};
__device__ __forceinline__ Rows k_rows(const GqaKv& kv) {
    return {(uint8_t*) kv.k, kv.q8, (size_t) kv.max_ctx * kv.n_kv};
}
__device__ __forceinline__ Rows v_rows(const GqaKv& kv) {
    return {(uint8_t*) kv.v, kv.q8, (size_t) kv.max_ctx * kv.n_kv};
}

// one row of 128 values into the cache (kD threads, value t from each): FP16, or INT8 with a scale per 32 (a warp)
__device__ __forceinline__ void store_row(const GqaKv& kv, uint8_t* base, size_t r, int t, float val) {
    if (kv.q8) {
        const float amax = wmax(fabsf(val));
        const float sc = amax / 127.0f;
        const int q = sc > 0.0f ? __float2int_rn(val / sc) : 0;
        base[r * kD + t] = (uint8_t) (int8_t) max(-127, min(127, q));
        if ((t & 31) == 0)
            ((__half*) (base + (size_t) kv.max_ctx * kv.n_kv * kD))[r * (kD / 32) + t / 32] = __float2half(sc);
    } else {
        ((__half*) base)[r * kD + t] = __float2half(val);
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
        store_row(kv, (uint8_t*) kv.v, (size_t) pos * n_kv + h, t, v[(size_t) row * kld + (size_t) h * kD + t]);
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
    const float o = t < kD / 2 ? x0 * c - x1 * s : x0 * s + x1 * c;
    if (is_q) q[(size_t) row * qld + (size_t) h * kD + t] = o;
    else store_row(kv, (uint8_t*) kv.k, (size_t) pos * n_kv + h, t, o);
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
    const int p0 = split * kChunk, np = min(kChunk, n_pos - p0);
    const Rows K = k_rows(kv), V = v_rows(kv);
    for (int i = t; i < G * kD; i += blockDim.x) sq[i] = q[(size_t) (kvh * G) * kD + i] * scale;
    __syncthreads();
    for (int j = w; j < np; j += 8) {
        float k0, k1, k2, k3;
        K.load4((size_t) (p0 + j) * n_kv + kvh, 4 * lane, k0, k1, k2, k3);
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
            const float e = __expf(sp[g * kChunk + j] - m);
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
            float v0, v1, v2, v3;
            V.load4((size_t) (p0 + j) * n_kv + kvh, d4, v0, v1, v2, v3);
            const float pj = sp[g * kChunk + j];
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
                                                         float* __restrict__ out) {
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
    out[(size_t) h * kD + t] = acc / den;
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
    const Rows K = k_rows(kv), V = v_rows(kv);
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
                K.load4((size_t) (k0 + r) * n_kv + kvh, d, a, b, c, e);
                V.load4((size_t) (k0 + r) * n_kv + kvh, d, va, vb, vc, ve);
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
    const Rows K = k_rows(kv), V = v_rows(kv);
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
            if (k0 + r <= last) K.load4((size_t) (k0 + r) * n_kv + kvh, d, a, b, c, e);
            __half2* kr = (__half2*) (sK + r * kLdQ + d);
            kr[0] = __floats2half2_rn(a, b);
            kr[1] = __floats2half2_rn(c, e);
            if (with_v) {
                if (k0 + r <= last) V.load4((size_t) (k0 + r) * n_kv + kvh, d, a, b, c, e);
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

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) std::fprintf(stderr, "glm_gqa %s: %s\n", what, cudaGetErrorString(e));
}

}  // namespace

size_t gqa_kv_layer_bytes(int n_kv, int64_t max_ctx, bool q8) {
    const size_t rows = (size_t) max_ctx * n_kv;
    return 2 * rows * (q8 ? (size_t) kD + (kD / 32) * sizeof(__half) : (size_t) kD * sizeof(__half));
}

GqaKv gqa_kv_view(void* layer_base, int n_kv, int64_t max_ctx, bool q8) {
    GqaKv kv;
    kv.k = layer_base;
    kv.v = (uint8_t*) layer_base + gqa_kv_layer_bytes(n_kv, max_ctx, q8) / 2;
    kv.q8 = q8 ? 1 : 0;
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
    const int n_split = (n_pos + kChunk - 1) / kChunk;
    float* part = scratch;
    float* out = scratch + (size_t) n_head * n_split * (kD + 2);
    gqa_attn_kernel<<<dim3(kv.n_kv, n_split), 256, 0, s>>>(q, kv, n_pos, n_head, 1.0f / sqrtf((float) kD), part);
    check("gqa_attn");
    gqa_combine_kernel<<<n_head, kD, 0, s>>>(part, n_split, out);
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
        return true;
    }
#endif
    gqa_attn_rows_kernel<<<dim3((T + kQT - 1) / kQT, n_head), 256, 0, s>>>(q, qld, kv, p0, T, n_head,
                                                                          1.0f / sqrtf((float) kD), (__half*) out16, old);
    check("gqa_attn_rows");
    return true;
}

}  // namespace strata::kernels::glmf
