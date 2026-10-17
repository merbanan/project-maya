// src/kernels/cuda/glm_dsv4.cu - DeepSeek-V4 Flash's attention for the Glm5Model fast path (one token): the cache
// update (raw row + compressors), the indexer's pick, the attention with sinks, the mHC head.  See glm_dsv4.hpp; the
// arithmetic follows antirez's ds4.c (the CPU reference) step by step.
#include "strata/kernels/glm_dsv4.hpp"

#include <cuda_fp16.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels::glmf {
namespace {

__device__ __forceinline__ float h2f(uint16_t v) { return __half2float(__ushort_as_half(v)); }
__device__ __forceinline__ uint16_t f2h(float v) { return __half_as_ushort(__float2half_rn(v)); }

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
// the sum over the whole block (every thread calls it; red: 32 floats of shared memory)
__device__ float bsum(float v, float* red) {
    v = wsum(v);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float t = 0.0f;
    for (int i = 0; i < (int) (blockDim.x >> 5); ++i) t += red[i];
    __syncthreads();
    return t;
}

// ---- the model's quantization round trips (ds4.c: dsv4_e4m3fn_dequant_cpu, dsv4_e2m1fn_dequant_cpu)
// E4M3FN: nearest of the 127 non-negative values up to 448, ties to the even mantissa
__device__ __forceinline__ float e4m3_rt(float x) {
    const float ax = fminf(fabsf(x), 448.0f);
    float q;
    if (ax < 0.015625f) {   // subnormals: steps of 2^-9
        q = rintf(ax * 512.0f) * 0.001953125f;
    } else {
        int e;
        frexpf(ax, &e);                         // ax = m 2^e, m in [0.5, 1): the step is 2^(e - 1 - 3)
        const float step = ldexpf(1.0f, e - 4);
        q = fminf(rintf(ax / step) * step, 448.0f);
    }
    return copysignf(q, x);
}
// E2M1: {0, .5, 1, 1.5, 2, 3, 4, 6}, ties to the even code
__device__ __forceinline__ float e2m1_rt(float x) {
    const float v[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float ax = fminf(fabsf(x), 6.0f);
    int best = 0;
    float bd = ax;
#pragma unroll
    for (int i = 1; i < 8; ++i) {
        const float d = fabsf(ax - v[i]);
        if (d < bd || (d == bd && (i & 1) == 0 && (best & 1) != 0)) {
            best = i;
            bd = d;
        }
    }
    return x < 0.0f ? -v[best] : v[best];
}

// the rope of pair i (dims 2i, 2i + 1 of the tail) at position pos (rope_tail_ext_inplace; inverse: sin negated)
__device__ __forceinline__ void rope_pair(float& x0, float& x1, int i, float pos, const Dsv4Rope& r, bool inverse) {
    const float extrap = pos * powf(r.base, -2.0f * (float) i / (float) kDsv4Rot);
    float theta = r.scale * extrap;
    if (r.ext != 0.0f) {
        const float y = ((float) i - r.corr0) / fmaxf(0.001f, r.corr1 - r.corr0);
        const float ramp = (1.0f - fminf(1.0f, fmaxf(0.0f, y))) * r.ext;
        theta = theta * (1.0f - ramp) + extrap * ramp;
    }
    float sn, cs;
    sincosf(theta, &sn, &cs);
    if (inverse) sn = -sn;
    const float a = x0, b = x1;
    x0 = a * cs - b * sn;
    x1 = a * sn + b * cs;
}

// v (thread d's dim of a 512-wide row in buf) -> the rope on dims 448..511 at pos; every thread of the block calls it
__device__ float rope_tail512(float v, float* buf, float pos, const Dsv4Rope& r, bool inverse) {
    const int d = threadIdx.x;
    buf[d] = v;
    __syncthreads();
    if (d >= kDsv4Hd - kDsv4Rot && ((d - (kDsv4Hd - kDsv4Rot)) & 1) == 0) {
        float x0 = buf[d], x1 = buf[d + 1];
        rope_pair(x0, x1, (d - (kDsv4Hd - kDsv4Rot)) >> 1, pos, r, inverse);
        buf[d] = x0;
        buf[d + 1] = x1;
    }
    __syncthreads();
    const float o = buf[d];
    __syncthreads();
    return o;
}

// the FP8 round trip of the 448 non-rope dims (64-wide blocks, a power-of-two scale) - 512-thread blocks, thread d
__device__ float fp8_448(float v, float* red16) {
    const int d = threadIdx.x;
    const float am = wmax(d < kDsv4Hd - kDsv4Rot ? fabsf(v) : 0.0f);
    if ((d & 31) == 0) red16[d >> 5] = am;
    __syncthreads();
    float out = v;
    if (d < kDsv4Hd - kDsv4Rot) {
        const float amax = fmaxf(fmaxf(red16[(d >> 6) * 2], red16[(d >> 6) * 2 + 1]), 1.0e-4f);
        const float scale = ldexpf(1.0f, (int) ceilf(log2f(amax / 448.0f)));
        out = e4m3_rt(fminf(fmaxf(v / scale, -448.0f), 448.0f)) * scale;
    }
    __syncthreads();
    return out;
}

// pool dim j of a compressor's window: softmax over the scores, per dim (compressor_pool_decode_state)
__device__ float pool_dim(const float* kv, const float* sc, int j, int hd, int ratio) {
    const int W = ratio == 4 ? 2 * hd : hd;
    float mx = -INFINITY;
    if (ratio == 4) {
        for (int r = 0; r < 4; ++r) mx = fmaxf(mx, fmaxf(sc[r * W + j], sc[(4 + r) * W + hd + j]));
    } else {
        for (int r = 0; r < ratio; ++r) mx = fmaxf(mx, sc[r * W + j]);
    }
    if (mx <= -INFINITY) return 0.0f;
    float den = 0.0f, sum = 0.0f;
    if (ratio == 4) {
        for (int r = 0; r < 4; ++r) {
            const float wp = expf(sc[r * W + j] - mx), wc = expf(sc[(4 + r) * W + hd + j] - mx);
            den += wp + wc;
            sum += wp * kv[r * W + j] + wc * kv[(4 + r) * W + hd + j];
        }
    } else {
        for (int r = 0; r < ratio; ++r) {
            const float w = expf(sc[r * W + j] - mx);
            den += w;
            sum += w * kv[r * W + j];
        }
    }
    return den > 0.0f ? sum / den : 0.0f;
}

// a compressor's window row for this position (both halves for ratio 4): its kv and score (+ the ape row)
__device__ void comp_store(float* st_kv, float* st_sc, const float* kv, const float* sc, const float* ape, int W,
                           int ratio, int pos) {
    const int pm = pos % ratio, row = ratio == 4 ? 4 + pm : pm;
    for (int j = threadIdx.x; j < W; j += blockDim.x) {
        st_kv[(size_t) row * W + j] = kv[j];
        st_sc[(size_t) row * W + j] = sc[j] + ape[(size_t) pm * W + j];
    }
}
// ratio 4: the window's current half becomes the previous one (after the pool read it)
__device__ void comp_shift4(float* st_kv, float* st_sc, int W) {
    for (int j = threadIdx.x; j < 4 * W; j += blockDim.x) {
        st_kv[j] = st_kv[4 * W + j];
        st_sc[j] = st_sc[4 * W + j];
    }
}

__device__ void comp_emit(float v, const float* norm, float eps, int ratio, int pos, bool idx, uint16_t* rows,
                          const Dsv4Rope& r, float* buf, float* red);
// one position of a compressor (compressor_decode_one): its window row, and at every ratio-th position the pooled
// row - norm, rope at pos + 1 - ratio, then FP8 (the attention's, hd 512) or Hadamard + FP4 (the indexer's, hd 128) -
// at index (pos + 1) / ratio - 1.  512-thread blocks; every thread calls it.
__device__ void comp_step(float* st_kv, float* st_sc, const float* kv, const float* sc, const float* ape,
                          const float* norm, float eps, int ratio, int pos, bool idx, uint16_t* rows, const Dsv4Rope& r,
                          float* buf, float* red) {
    const int d = threadIdx.x;
    const int hd = idx ? kDsv4Ihd : kDsv4Hd, W = ratio == 4 ? 2 * hd : hd;
    comp_store(st_kv, st_sc, kv, sc, ape, W, ratio, pos);
    __syncthreads();
    if ((pos + 1) % ratio != 0) return;
    const int at = (pos + 1) / ratio - 1;
    const bool on = d < hd;
    float v = on ? pool_dim(st_kv, st_sc, d, hd, ratio) : 0.0f;
    __syncthreads();
    if (ratio == 4) comp_shift4(st_kv, st_sc, W);
    comp_emit(v, norm, eps, ratio, pos, idx, rows, r, buf, red);
}

// a pooled row's end (thread d holds dim d of the pool, 0 past hd): norm, rope at pos + 1 - ratio, then FP8 (the
// attention's) or Hadamard + FP4 (the indexer's), at index (pos + 1) / ratio - 1
__device__ void comp_emit(float v, const float* norm, float eps, int ratio, int pos, bool idx, uint16_t* rows,
                          const Dsv4Rope& r, float* buf, float* red) {
    const int d = threadIdx.x;
    const int hd = idx ? kDsv4Ihd : kDsv4Hd;
    const int at = (pos + 1) / ratio - 1;
    const bool on = d < hd;
    const float ss = bsum(v * v, red);
    v = on ? v * rsqrtf(ss / (float) hd + eps) * norm[d] : 0.0f;
    if (!idx) {
        v = rope_tail512(v, buf, (float) (pos + 1 - ratio), r, false);
        v = fp8_448(v, red);
        rows[(size_t) at * kDsv4Hd + d] = f2h(v);
        __syncthreads();
        return;
    }
    if (on) buf[d] = v;
    __syncthreads();
    // rope: dims 64..127 of the 128
    if (on && d >= kDsv4Ihd - kDsv4Rot && ((d - (kDsv4Ihd - kDsv4Rot)) & 1) == 0) {
        float x0 = buf[d], x1 = buf[d + 1];
        rope_pair(x0, x1, (d - (kDsv4Ihd - kDsv4Rot)) >> 1, (float) (pos + 1 - 4), r, false);
        buf[d] = x0;
        buf[d + 1] = x1;
    }
    __syncthreads();
    // Hadamard 128 (butterflies, then 1/sqrt(128))
    for (int st = 1; st < kDsv4Ihd; st <<= 1) {
        if (d < kDsv4Ihd / 2) {
            const int b = (d / st) * 2 * st + d % st;
            const float x0 = buf[b], x1 = buf[b + st];
            buf[b] = x0 + x1;
            buf[b + st] = x0 - x1;
        }
        __syncthreads();
    }
    if (on) {
        float x = buf[d] * 0.08838834764831845f;
        // FP4 in 32-blocks (one warp each)
        const float am = fmaxf(wmax(fabsf(x)), 7.052966104933725e-38f);
        const float scale = ldexpf(1.0f, (int) ceilf(log2f(am / 6.0f)));
        x = e2m1_rt(fminf(fmaxf(x / scale, -6.0f), 6.0f)) * scale;
        rows[(size_t) at * kDsv4Ihd + d] = f2h(x);
    }
    __syncthreads();
}

// a raw K/V row (norm, rope at pos, FP8) - 512 threads, thread d
__device__ float raw_row(float v, const float* w, float eps, int pos, const Dsv4Rope& r, float* buf, float* red) {
    const float ss = bsum(v * v, red);
    v = v * rsqrtf(ss / (float) kDsv4Hd + eps) * w[threadIdx.x];
    v = rope_tail512(v, buf, (float) pos, r, false);
    return fp8_448(v, red);
}

// block 0: the raw row; block 1: the attention compressor (ratio > 0); block 2: the indexer compressor (ratio 4).
// 512 threads.
__global__ void __launch_bounds__(512) kv_update_kernel(const Dsv4KvArgs a, const Dsv4Layer L, const Dsv4Rope r,
                                                        int pos) {
    __shared__ float buf[kDsv4Hd];
    __shared__ float red[32];
    const int d = threadIdx.x;
    if (blockIdx.x == 0) {
        const float v = raw_row(a.kv_raw[d], a.kv_norm, a.eps, pos, r, buf, red);
        L.swa[(size_t) (pos % a.n_swa) * kDsv4Hd + d] = f2h(v);
        return;
    }
    if (blockIdx.x == 1)
        comp_step(L.st_kv, L.st_sc, a.comp_kv, a.comp_sc, a.comp_ape, a.comp_norm, a.eps, L.ratio, pos, false, L.comp,
                  r, buf, red);
    else
        comp_step(L.ist_kv, L.ist_sc, a.icomp_kv, a.icomp_sc, a.icomp_ape, a.icomp_norm, a.eps, 4, pos, true, L.icomp,
                  r, buf, red);
}

__global__ void __launch_bounds__(1024) qr_norm_kernel(const float* qr, const float* w, float eps, int n, float* out) {
    __shared__ float red[32];
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += qr[i] * qr[i];
    ss = bsum(ss, red);
    const float inv = rsqrtf(ss / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) out[i] = qr[i] * inv * w[i];
}

// ---- the indexer: per head rope + Hadamard + FP4 (in place), then the scores, then the top k
__global__ void __launch_bounds__(kDsv4Ihd) idx_q_kernel(float* q, Dsv4Rope r, int pos) {
    __shared__ float buf[kDsv4Ihd];
    const int d = threadIdx.x;
    float* qh = q + (size_t) blockIdx.x * kDsv4Ihd;
    buf[d] = qh[d];
    __syncthreads();
    if (d >= kDsv4Ihd - kDsv4Rot && ((d - (kDsv4Ihd - kDsv4Rot)) & 1) == 0) {
        float x0 = buf[d], x1 = buf[d + 1];
        rope_pair(x0, x1, (d - (kDsv4Ihd - kDsv4Rot)) >> 1, (float) pos, r, false);
        buf[d] = x0;
        buf[d + 1] = x1;
    }
    __syncthreads();
    for (int st = 1; st < kDsv4Ihd; st <<= 1) {
        if (d < kDsv4Ihd / 2) {
            const int b = (d / st) * 2 * st + d % st;
            const float x0 = buf[b], x1 = buf[b + st];
            buf[b] = x0 + x1;
            buf[b + st] = x0 - x1;
        }
        __syncthreads();
    }
    float x = buf[d] * 0.08838834764831845f;
    const float am = fmaxf(wmax(fabsf(x)), 7.052966104933725e-38f);
    const float scale = ldexpf(1.0f, (int) ceilf(log2f(am / 6.0f)));
    qh[d] = e2m1_rt(fminf(fmaxf(x / scale, -6.0f), 6.0f)) * scale;
}

constexpr int kIdxRows = 16;   // pooled rows a score block
// score[c] = sum_h relu(q_h . k_c) w_h / sqrt(128 n_head); n_head = 64 (256 threads: head t & 63, rows t >> 6 + 4 j)
__global__ void __launch_bounds__(256) idx_score_kernel(const float* q, const float* w, const uint16_t* k, int n_comp,
                                                        float wscale, float* score) {
    __shared__ float sq[64][kDsv4Ihd + 1];
    __shared__ float sk[kIdxRows][kDsv4Ihd];
    __shared__ float sp[kIdxRows][64 + 1];
    const int t = threadIdx.x, c0 = blockIdx.x * kIdxRows;
    for (int i = t; i < 64 * kDsv4Ihd; i += 256) sq[i / kDsv4Ihd][i % kDsv4Ihd] = q[i];
    for (int i = t; i < kIdxRows * kDsv4Ihd; i += 256) {
        const int r = i / kDsv4Ihd;
        sk[r][i % kDsv4Ihd] = c0 + r < n_comp ? h2f(k[(size_t) (c0 + r) * kDsv4Ihd + i % kDsv4Ihd]) : 0.0f;
    }
    __syncthreads();
    const int h = t & 63;
    const float wh = w[h] * wscale;
    for (int r = t >> 6; r < kIdxRows; r += 4) {
        float s = 0.0f;
#pragma unroll 8
        for (int i = 0; i < kDsv4Ihd; ++i) s += sq[h][i] * sk[r][i];
        sp[r][h] = fmaxf(s, 0.0f) * wh;
    }
    __syncthreads();
    const int lane = t & 31, wp = t >> 5;
    for (int r = wp; r < kIdxRows; r += 8) {
        const float s = wsum(sp[r][lane] + sp[r][lane + 32]);
        if (lane == 0 && c0 + r < n_comp) score[c0 + r] = s;
    }
}

// The same on the tensor cores: q and the pooled keys are FP4 codes times a power of two, exact in FP16, so the
// products are exact and only the F32 summation order differs.  Block: 4 warps of 16 rows; a warp stages its rows
// (4-byte loads: the rows are only 4-byte aligned in the state), S (16 rows x 64 heads) = K . Q^T by WMMA, then
// relu * w summed over the heads, a row per lane pair... a warp reduction a row.
constexpr int kTcRows = 64, kTcLdK = kDsv4Ihd + 8, kTcLdS = 64 + 4;
__global__ void __launch_bounds__(128) idx_score_tc_kernel(const float* q, const float* w, const uint16_t* k,
                                                           int n_comp, float wscale, float* score) {
    using namespace nvcuda;
    __shared__ __align__(32) __half sq[64 * kTcLdK];
    static_assert(16 * kTcLdK * 2 == 16 * kTcLdS * 4, "a warp's key tile and score tile share their buffer");
    __shared__ __align__(32) unsigned char buf[4][16 * kTcLdK * 2];   // the warp's K rows, then its S rows
    const int t = threadIdx.x, lane = t & 31, wp = t >> 5;
    for (int i = t; i < 64 * kDsv4Ihd; i += 128) sq[(i / kDsv4Ihd) * kTcLdK + i % kDsv4Ihd] = __float2half(q[i]);
    const int r0 = blockIdx.x * kTcRows + wp * 16;
    __half* K = (__half*) buf[wp];
    float* S = (float*) buf[wp];
    const uint32_t* k32 = (const uint32_t*) k;
    for (int i = lane; i < 16 * kDsv4Ihd / 2; i += 32) {   // 16 rows x 64 words
        const int r = i / (kDsv4Ihd / 2), wd = i % (kDsv4Ihd / 2);
        const uint32_t v = r0 + r < n_comp ? k32[(size_t) (r0 + r) * (kDsv4Ihd / 2) + wd] : 0u;
        *(uint32_t*) (K + r * kTcLdK + 2 * wd) = v;
    }
    __syncthreads();
    if (r0 >= n_comp) return;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4];
#pragma unroll
    for (int n = 0; n < 4; ++n) wmma::fill_fragment(acc[n], 0.0f);
#pragma unroll
    for (int kk = 0; kk < kDsv4Ihd / 16; ++kk) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
        wmma::load_matrix_sync(fa, K + kk * 16, kTcLdK);
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
            wmma::load_matrix_sync(fb, sq + (n * 16) * kTcLdK + kk * 16, kTcLdK);
            wmma::mma_sync(acc[n], fa, fb, acc[n]);
        }
    }
    __syncwarp();   // (every lane's K fragments are loaded)
#pragma unroll
    for (int n = 0; n < 4; ++n) wmma::store_matrix_sync(S + n * 16, acc[n], kTcLdS, wmma::mem_row_major);
    __syncwarp();
    const float w0 = w[lane] * wscale, w1 = w[lane + 32] * wscale;
    for (int r = 0; r < 16 && r0 + r < n_comp; ++r) {
        const float v = wsum(fmaxf(S[r * kTcLdS + lane], 0.0f) * w0 + fmaxf(S[r * kTcLdS + lane + 32], 0.0f) * w1);
        if (lane == 0) score[r0 + r] = v;
    }
}

// the top k of n scores (any order): a radix select on the order-preserving keys (3 passes of 11 / 11 / 10 bits), then
// every key above the threshold and the lowest-indexed equal ones.  One block of 1024 threads.
__device__ __forceinline__ uint32_t okey(float f) {
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);   // larger float -> larger key
}
// a block of 1024 threads: the top k of n scores into sel (the decode's and the prompt rows' selection)
// an inclusive prefix sum over a 1024-thread block: warp shuffles, then the 32 warp totals (3 barriers, not 20)
__device__ __forceinline__ int block_incl_scan(int v, int* wsum) {
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int n = __shfl_up_sync(0xffffffffu, v, o);
        if (lane >= o) v += n;
    }
    if (lane == 31) wsum[wid] = v;
    __syncthreads();
    if (wid == 0) {
        int w = wsum[lane];
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int n = __shfl_up_sync(0xffffffffu, w, o);
            if (lane >= o) w += n;
        }
        wsum[lane] = w;
    }
    __syncthreads();
    if (wid > 0) v += wsum[wid - 1];
    __syncthreads();   // (wsum free for the next scan)
    return v;
}
__device__ __forceinline__ void topk_select(const float* score, int n, int k, int* sel) {
    __shared__ unsigned int hist[2048];
    __shared__ uint32_t s_prefix, s_mask;
    __shared__ int s_need, s_bin;
    __shared__ int s_out, s_eq;
    const int t = threadIdx.x;
    if (t == 0) {
        s_prefix = 0u;
        s_mask = 0u;
        s_need = k;
    }
    __syncthreads();
    const int shifts[3] = {21, 10, 0}, bits[3] = {11, 11, 10};
    for (int pass = 0; pass < 3; ++pass) {
        const int sh = shifts[pass], nb = 1 << bits[pass];
        for (int i = t; i < nb; i += 1024) hist[i] = 0u;
        __syncthreads();
        const uint32_t pre = s_prefix, msk = s_mask;
        for (int i = t; i < n; i += 1024) {
            const uint32_t key = okey(score[i]);
            if ((key & msk) == pre) atomicAdd(&hist[(key >> sh) & (uint32_t) (nb - 1)], 1u);
        }
        __syncthreads();
        // from the top bin down, the bin where the count reaches s_need: suffix sums of the bins (a block scan over
        // the bins in descending order, two a thread), then the one thread whose bin crosses it
        {
            __shared__ int s_suf[1024];
            const int per = nb / 1024;   // 2 or 1 bins a thread (nb 2048 / 1024)
            const int need0 = s_need;    // (read before any thread can change it below)
            int c = 0;
            for (int j = 0; j < per; ++j) c += (int) hist[nb - 1 - (t * per + j)];
            const int incl = block_incl_scan(c, s_suf);
            int above = incl - c;   // the keys in the bins above this thread's
            for (int j = 0; j < per; ++j) {
                const int b = nb - 1 - (t * per + j), h = (int) hist[b];
                if (above < need0 && above + h >= need0) {   // (exactly one bin; bin 0 takes the rest)
                    s_bin = b;
                    s_need = need0 - above;
                    s_prefix = pre | ((uint32_t) b << sh);
                    s_mask = msk | ((uint32_t) (nb - 1) << sh);
                }
                above += h;
            }
            __syncthreads();
        }
    }
    // the threshold key is s_prefix; s_need of the keys equal to it are taken (the lowest indices)
    const uint32_t thr = s_prefix;
    if (t == 0) {
        s_out = 0;
        s_eq = 0;
    }
    __syncthreads();
    for (int i = t; i < n; i += 1024)
        if (okey(score[i]) > thr) sel[atomicAdd(&s_out, 1)] = i;
    __syncthreads();
    // the equal ones in index order: a thread a contiguous chunk, a block prefix sum of their counts, then each thread
    // writes the ones that still fit (was one thread over all n: most of the indexer's time)
    __shared__ int s_cnt[1024];
    const int chunk = (n + 1023) / 1024, lo = t * chunk, hi = min(n, lo + chunk);
    int cnt = 0;
    for (int i = lo; i < hi; ++i) cnt += okey(score[i]) == thr;
    int before = block_incl_scan(cnt, s_cnt) - cnt;
    const int need = s_need, base = s_out;
    for (int i = lo; i < hi && before < need; ++i)
        if (okey(score[i]) == thr) sel[base + before++] = i;
}
__global__ void __launch_bounds__(1024) topk_kernel(const float* score, int n, int k, int* sel) {
    topk_select(score, n, k, sel);
}

// ---- the attention: one block a head (256 threads), the scores in dynamic shared memory
__global__ void __launch_bounds__(256) attn_kernel(const float* q, const float* sinks, const uint16_t* swa,
                                                   const uint16_t* comp, int n_swa, int pos, const int* sel,
                                                   int n_sel, Dsv4Rope r, float eps, float* out) {
    extern __shared__ float sc[];
    __shared__ float sq[kDsv4Hd];
    __shared__ float red[32];
    __shared__ float so[kDsv4Hd];
    const int h = blockIdx.x, t = threadIdx.x, lane = t & 31, wp = t >> 5;
    // q: the per-head RMS norm (no weight), then the rope of the tail
    const float* qh = q + (size_t) h * kDsv4Hd;
    float a0 = qh[t], a1 = qh[t + 256];
    const float ss = bsum(a0 * a0 + a1 * a1, red);
    const float inv = rsqrtf(ss / (float) kDsv4Hd + eps);
    sq[t] = a0 * inv;
    sq[t + 256] = a1 * inv;
    __syncthreads();
    if (t < kDsv4Rot / 2) {
        const int d = kDsv4Hd - kDsv4Rot + 2 * t;
        float x0 = sq[d], x1 = sq[d + 1];
        rope_pair(x0, x1, t, (float) pos, r, false);
        sq[d] = x0;
        sq[d + 1] = x1;
    }
    __syncthreads();
    const int n_raw = min(pos + 1, n_swa), n_all = n_raw + n_sel;
    const float scale = rsqrtf((float) kDsv4Hd);
    const auto row = [&](int i) -> const uint16_t* {
        if (i < n_raw) return swa + (size_t) ((pos - n_raw + 1 + i) % n_swa) * kDsv4Hd;
        const int c = i - n_raw;
        return comp + (size_t) (sel != nullptr ? sel[c] : c) * kDsv4Hd;
    };
    // the scores: a warp a row, lane l dims l + 32 j
    for (int i = wp; i < n_all; i += 8) {
        const uint16_t* kr = row(i);
        float s = 0.0f;
#pragma unroll
        for (int j = 0; j < kDsv4Hd / 32; ++j) s += sq[lane + 32 * j] * h2f(kr[lane + 32 * j]);
        s = wsum(s);
        if (lane == 0) sc[i] = s * scale;
    }
    __syncthreads();
    float mx = sinks[h];
    for (int i = t; i < n_all; i += 256) mx = fmaxf(mx, sc[i]);
    mx = wmax(mx);
    if (lane == 0) red[wp] = mx;
    __syncthreads();
    mx = red[0];
    for (int i = 1; i < 8; ++i) mx = fmaxf(mx, red[i]);
    __syncthreads();
    float den = 0.0f;
    for (int i = t; i < n_all; i += 256) {
        const float w = expf(sc[i] - mx);
        sc[i] = w;
        den += w;
    }
    den = bsum(den, red) + expf(sinks[h] - mx);
    // the output: a warp every 8th row (lane l dims l + 32 j, the loads coalesced), the 8 partial sums added in
    // shared memory - the rows were one serial chain of loads a thread before
    __syncthreads();   // (sc complete)
    float acc[kDsv4Hd / 32];
#pragma unroll
    for (int j = 0; j < kDsv4Hd / 32; ++j) acc[j] = 0.0f;
    for (int i = wp; i < n_all; i += 8) {
        const uint16_t* kr = row(i);
        const float w = sc[i];
#pragma unroll
        for (int j = 0; j < kDsv4Hd / 32; ++j) acc[j] += w * h2f(kr[lane + 32 * j]);
    }
    float* part = sc + ((n_all + 31) & ~31);   // 8 x kDsv4Hd floats after the scores (dynamic shared memory)
#pragma unroll
    for (int j = 0; j < kDsv4Hd / 32; ++j) part[wp * kDsv4Hd + lane + 32 * j] = acc[j];
    __syncthreads();
    const float id = 1.0f / den;
    for (int d = t; d < kDsv4Hd; d += 256) {
        float v = 0.0f;
#pragma unroll
        for (int w8 = 0; w8 < 8; ++w8) v += part[w8 * kDsv4Hd + d];
        so[d] = v * id;
    }
    __syncthreads();
    if (t < kDsv4Rot / 2) {   // the output's rope undone (the tail)
        const int d = kDsv4Hd - kDsv4Rot + 2 * t;
        float x0 = so[d], x1 = so[d + 1];
        rope_pair(x0, x1, t, (float) pos, r, true);
        so[d] = x0;
        so[d + 1] = x1;
    }
    __syncthreads();
    out[(size_t) h * kDsv4Hd + t] = so[t];
    out[(size_t) h * kDsv4Hd + t + 256] = so[t + 256];
}

// ---- the mHC head (output_hc_head_one) and output_norm: one block of 1024 threads
__global__ void __launch_bounds__(1024) hc_head_kernel(const float* R, const uint16_t* fn, const float* scale,
                                                       const float* base, float hc_eps, const float* norm_w, float eps,
                                                       int n_embd, float* x) {
    __shared__ float red[32];
    __shared__ float pre[4];
    const int t = threadIdx.x, hd = 4 * n_embd;
    float ss = 0.0f;
    for (int i = t; i < hd; i += 1024) ss += R[i] * R[i];
    ss = bsum(ss, red);
    const float inv = rsqrtf(ss / (float) hd + eps);
    for (int m = 0; m < 4; ++m) {
        float acc = 0.0f;
        for (int i = t; i < hd; i += 1024) acc += h2f(fn[(size_t) m * hd + i]) * R[i] * inv;
        acc = bsum(acc, red);
        if (t == 0) pre[m] = 1.0f / (1.0f + expf(-(acc * scale[0] + base[m]))) + hc_eps;
    }
    __syncthreads();
    float s2 = 0.0f;
    for (int e = t; e < n_embd; e += 1024) {
        const float v = pre[0] * R[e] + pre[1] * R[n_embd + e] + pre[2] * R[2 * n_embd + e] + pre[3] * R[3 * n_embd + e];
        x[e] = v;
        s2 += v * v;
    }
    s2 = bsum(s2, red);
    const float i2 = rsqrtf(s2 / (float) n_embd + eps);
    for (int e = t; e < n_embd; e += 1024) x[e] = x[e] * i2 * norm_w[e];
}

// ================================================================ the prompt path: a sub-batch of T rows
// qr = rms_norm(qr_raw) * w per row, as F32 and FP16 (T blocks of 1024)
__global__ void __launch_bounds__(1024) rows_qr_kernel(const float* qr_raw, const float* w, float eps, int n, float* qr,
                                                      uint16_t* qr16) {
    __shared__ float red[32];
    const size_t o = (size_t) blockIdx.x * n;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += qr_raw[o + i] * qr_raw[o + i];
    ss = bsum(ss, red);
    const float inv = rsqrtf(ss / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = qr_raw[o + i] * inv * w[i];
        qr[o + i] = v;
        qr16[o + i] = f2h(v);
    }
}
// the raw K/V rows of the sub-batch (T blocks of 512): norm, rope, FP8 -> raw16
__global__ void __launch_bounds__(512) rows_kv_kernel(const float* kv_raw, const float* w, float eps, int p0, Dsv4Rope r,
                                                     uint16_t* raw16) {
    __shared__ float buf[kDsv4Hd];
    __shared__ float red[32];
    const int t = blockIdx.x, d = threadIdx.x;
    const float v = raw_row(kv_raw[(size_t) t * kDsv4Hd + d], w, eps, p0 + t, r, buf, red);
    raw16[(size_t) t * kDsv4Hd + d] = f2h(v);
}
// ---- the compressor over a sub-batch in parallel: a block (512) a pooled row the sub-batch completes. A window entry
// comes from the sub-batch's rows (position q >= pt: kv / sc + ape) or, before pt, from the state the serial steps
// left (ratio 4: the current half's rows 4 + r from c_old = pt - pt % 4 on, the previous half's rows r before it).
// Same arithmetic, same order as pool_dim: the same rows. Then comp_state_kernel leaves the state the serial steps
// would have.
struct CompSrc {
    const float *kv, *sc, *ape, *st_kv, *st_sc;
    int W, ratio, pt, c_old;
};
__device__ __forceinline__ void comp_entry(const CompSrc& S, int q, int col, int st_row, float& kv, float& sc) {
    if (q >= S.pt) {
        kv = S.kv[(size_t) (q - S.pt) * S.W + col];
        sc = S.sc[(size_t) (q - S.pt) * S.W + col] + S.ape[(size_t) (q % S.ratio) * S.W + col];
    } else {
        kv = S.st_kv[(size_t) st_row * S.W + col];
        sc = S.st_sc[(size_t) st_row * S.W + col];
    }
}
__device__ float pool_dim_src(const CompSrc& S, int at, int j, int hd) {
    float mx = -INFINITY;
    float kp[4], sp[4], kc[4], scc[4];
    const int ratio = S.ratio;
    if (ratio == 4) {
        for (int r = 0; r < 4; ++r) {
            const int qp = (at - 1) * 4 + r, qc = at * 4 + r;
            comp_entry(S, qp, j, qp >= S.c_old ? 4 + r : r, kp[r], sp[r]);
            comp_entry(S, qc, hd + j, 4 + r, kc[r], scc[r]);
            mx = fmaxf(mx, fmaxf(sp[r], scc[r]));
        }
        if (mx <= -INFINITY) return 0.0f;
        float den = 0.0f, sum = 0.0f;
        for (int r = 0; r < 4; ++r) {
            const float wp = expf(sp[r] - mx), wc = expf(scc[r] - mx);
            den += wp + wc;
            sum += wp * kp[r] + wc * kc[r];
        }
        return den > 0.0f ? sum / den : 0.0f;
    }
    for (int r = 0; r < ratio; ++r) {
        float k, sv;
        comp_entry(S, at * ratio + r, j, r, k, sv);
        mx = fmaxf(mx, sv);
    }
    if (mx <= -INFINITY) return 0.0f;
    float den = 0.0f, sum = 0.0f;
    for (int r = 0; r < ratio; ++r) {
        float k, sv;
        comp_entry(S, at * ratio + r, j, r, k, sv);
        const float w = expf(sv - mx);
        den += w;
        sum += w * k;
    }
    return den > 0.0f ? sum / den : 0.0f;
}
__global__ void __launch_bounds__(512) rows_comp_par_kernel(CompSrc S, int first_at, const float* norm, float eps,
                                                           bool idx, uint16_t* rows, Dsv4Rope r) {
    __shared__ float buf[kDsv4Hd];
    __shared__ float red[32];
    const int at = first_at + blockIdx.x, hd = idx ? kDsv4Ihd : kDsv4Hd, d = threadIdx.x;
    const float v = d < hd ? pool_dim_src(S, at, d, hd) : 0.0f;
    comp_emit(v, norm, eps, S.ratio, (at + 1) * S.ratio - 1, idx, rows, r, buf, red);
}
// the state after the sub-batch's last position e - 1 (c_new = e - e % ratio): ratio 4 - when a window completed in
// the batch (c_new > pt), rows r := the window [c_new - 4, c_new) (from the batch, or the state's rows 4 + r before
// pt); then rows (4 +) q % ratio := the batch's positions [max(pt, c_new), e)
__global__ void __launch_bounds__(512) comp_state_kernel(CompSrc S, int e, float* st_kv, float* st_sc) {
    const int ratio = S.ratio, W = S.W, c_new = e - e % ratio;
    if (ratio == 4 && c_new > S.pt) {
        for (int i = threadIdx.x; i < 4 * W; i += blockDim.x) {
            const int r = i / W, col = i % W, q = c_new - 4 + r;
            float k, sv;
            comp_entry(S, q, col, 4 + r, k, sv);
            __syncwarp();
            st_kv[i] = k;
            st_sc[i] = sv;
        }
        __syncthreads();
    }
    const int q0 = max(S.pt, c_new), base = ratio == 4 ? 4 : 0;
    for (int i = threadIdx.x; i < (e - q0) * W; i += blockDim.x) {
        const int q = q0 + i / W, col = i % W;
        float k, sv;
        comp_entry(S, q, col, 0, k, sv);   // (q >= pt: from the batch)
        st_kv[(size_t) (base + q % ratio) * W + col] = k;
        st_sc[(size_t) (base + q % ratio) * W + col] = sv;
    }
}

// a compressor over the sub-batch, position after position (one block of 512): the decode's comp_step on each row
__global__ void __launch_bounds__(512) rows_comp_kernel(const float* kv, const float* sc, int T, int p0, const float* ape,
                                                       const float* norm, float eps, float* st_kv, float* st_sc,
                                                       int ratio, bool idx, uint16_t* rows, Dsv4Rope r) {
    __shared__ float buf[kDsv4Hd];
    __shared__ float red[32];
    const int W = ratio == 4 ? 2 * (idx ? kDsv4Ihd : kDsv4Hd) : (idx ? kDsv4Ihd : kDsv4Hd);
    for (int t = 0; t < T; ++t)
        comp_step(st_kv, st_sc, kv + (size_t) t * W, sc + (size_t) t * W, ape, norm, eps, ratio, p0 + t, idx, rows, r,
                  buf, red);
}
// the indexer's q of row t, head h (T x 64 blocks of 128): rope at p0 + t, Hadamard, FP4 - in place
__global__ void __launch_bounds__(kDsv4Ihd) rows_idx_q_kernel(float* q, int n_head, int p0, Dsv4Rope r) {
    __shared__ float buf[kDsv4Ihd];
    const int d = threadIdx.x, t = blockIdx.x / n_head;
    float* qh = q + (size_t) blockIdx.x * kDsv4Ihd;
    buf[d] = qh[d];
    __syncthreads();
    if (d >= kDsv4Ihd - kDsv4Rot && ((d - (kDsv4Ihd - kDsv4Rot)) & 1) == 0) {
        float x0 = buf[d], x1 = buf[d + 1];
        rope_pair(x0, x1, (d - (kDsv4Ihd - kDsv4Rot)) >> 1, (float) (p0 + t), r, false);
        buf[d] = x0;
        buf[d + 1] = x1;
    }
    __syncthreads();
    for (int st = 1; st < kDsv4Ihd; st <<= 1) {
        if (d < kDsv4Ihd / 2) {
            const int b = (d / st) * 2 * st + d % st;
            const float x0 = buf[b], x1 = buf[b + st];
            buf[b] = x0 + x1;
            buf[b + st] = x0 - x1;
        }
        __syncthreads();
    }
    float x = buf[d] * 0.08838834764831845f;
    const float am = fmaxf(wmax(fabsf(x)), 7.052966104933725e-38f);
    const float scale = ldexpf(1.0f, (int) ceilf(log2f(am / 6.0f)));
    qh[d] = e2m1_rt(fminf(fmaxf(x / scale, -6.0f), 6.0f)) * scale;
}
// the indexer's scores of row t over its visible pooled rows ((p0 + t + 1) / 4 of them): grid (T, row blocks of
// kIdxRowsB), 256 threads; 64 heads
constexpr int kIdxRowsB = 256;
__global__ void __launch_bounds__(256) rows_idx_score_kernel(const float* q, const float* w, const uint16_t* k, int p0,
                                                            float wscale, float* score, int ld) {
    __shared__ float sq[64][kDsv4Ihd + 1];
    __shared__ float sk[kIdxRows][kDsv4Ihd];
    __shared__ float sp[kIdxRows][64 + 1];
    const int t = blockIdx.x, tid = threadIdx.x;
    const int n_vis = (p0 + t + 1) / 4, c_lo = blockIdx.y * kIdxRowsB;
    if (c_lo >= n_vis) return;
    const int c_hi = min(n_vis, c_lo + kIdxRowsB);
    const float* qt = q + (size_t) t * 64 * kDsv4Ihd;
    for (int i = tid; i < 64 * kDsv4Ihd; i += 256) sq[i / kDsv4Ihd][i % kDsv4Ihd] = qt[i];
    const int h = tid & 63;
    const float wh = w[(size_t) t * 64 + h] * wscale;
    __syncthreads();
    for (int c0 = c_lo; c0 < c_hi; c0 += kIdxRows) {
        for (int i = tid; i < kIdxRows * kDsv4Ihd; i += 256) {
            const int r = i / kDsv4Ihd;
            sk[r][i % kDsv4Ihd] = c0 + r < c_hi ? h2f(k[(size_t) (c0 + r) * kDsv4Ihd + i % kDsv4Ihd]) : 0.0f;
        }
        __syncthreads();
        for (int r = tid >> 6; r < kIdxRows; r += 4) {
            float sv = 0.0f;
#pragma unroll 8
            for (int i = 0; i < kDsv4Ihd; ++i) sv += sq[h][i] * sk[r][i];
            sp[r][h] = fmaxf(sv, 0.0f) * wh;
        }
        __syncthreads();
        const int lane = tid & 31, wp = tid >> 5;
        for (int r = wp; r < kIdxRows; r += 8) {
            const float v = wsum(sp[r][lane] + sp[r][lane + 32]);
            if (lane == 0 && c0 + r < c_hi) score[(size_t) t * ld + c0 + r] = v;
        }
        __syncthreads();
    }
}
// the same scores on tensor cores: a block (row t, kIdxRowsB pooled rows) - S = Q_t (64 heads x 128) . K tile^T
// (wmma FP16 in, FP32 out; q is FP4-rounded, k FP16, so the products are the scalar kernel's), then per pooled row
// sum_h relu(S) w_h. Tiles of 64 pooled rows.
constexpr int kTcLd = kDsv4Ihd + 8, kTcSld = 64 + 4;
__global__ void __launch_bounds__(256) rows_idx_score_tc_kernel(const float* q, const float* w, const uint16_t* k, int p0,
                                                               float wscale, float* score, int ld) {
    using namespace nvcuda;
    extern __shared__ __align__(32) unsigned char smem_tc[];
    __half* sq = (__half*) smem_tc;                    // 64 x kTcLd
    __half* sk = sq + 64 * kTcLd;                      // 64 x kTcLd
    float* ss = (float*) (sk + 64 * kTcLd);            // 64 heads x kTcSld
    float* sw = ss + 64 * kTcSld;                      // 64 head weights
    float* sr = sw + 64;                               // 4 x 64 partial sums
    const int t = blockIdx.x, tid = threadIdx.x, wp = tid >> 5;
    const int n_vis = (p0 + t + 1) / 4, c_lo = blockIdx.y * kIdxRowsB;
    if (c_lo >= n_vis) return;
    const int c_hi = min(n_vis, c_lo + kIdxRowsB);
    const float* qt = q + (size_t) t * 64 * kDsv4Ihd;
    for (int i = tid; i < 64 * kDsv4Ihd; i += 256) sq[(i / kDsv4Ihd) * kTcLd + i % kDsv4Ihd] = __float2half(qt[i]);
    if (tid < 64) sw[tid] = w[(size_t) t * 64 + tid] * wscale;
    const __half* kh = (const __half*) k;
    for (int c0 = c_lo; c0 < c_hi; c0 += 64) {
        __syncthreads();
        for (int i = tid; i < 64 * (kDsv4Ihd / 8); i += 256) {   // 16-byte loads
            const int r = i / (kDsv4Ihd / 8), cc = (i % (kDsv4Ihd / 8)) * 8;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (c0 + r < c_hi) v = *(const uint4*) (kh + (size_t) (c0 + r) * kDsv4Ihd + cc);
            *(uint4*) (sk + r * kTcLd + cc) = v;
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < 2; ++j) {   // warp wp: output tiles 2 wp, 2 wp + 1 of the 4 x 4
            const int tile = 2 * wp + j, ti = tile >> 2, tj = tile & 3;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::fill_fragment(acc, 0.0f);
#pragma unroll
            for (int kk = 0; kk < kDsv4Ihd; kk += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
                wmma::load_matrix_sync(a, sq + ti * 16 * kTcLd + kk, kTcLd);
                wmma::load_matrix_sync(b, sk + tj * 16 * kTcLd + kk, kTcLd);
                wmma::mma_sync(acc, a, b, acc);
            }
            wmma::store_matrix_sync(ss + ti * 16 * kTcSld + tj * 16, acc, kTcSld, wmma::mem_row_major);
        }
        __syncthreads();
        {   // pooled row c = tid & 63, heads 16 (tid >> 6) .. + 15
            const int c = tid & 63, h0 = (tid >> 6) * 16;
            float v = 0.0f;
#pragma unroll
            for (int h = h0; h < h0 + 16; ++h) v += fmaxf(ss[h * kTcSld + c], 0.0f) * sw[h];
            sr[(tid >> 6) * 64 + c] = v;
        }
        __syncthreads();
        if (tid < 64 && c0 + tid < c_hi)
            score[(size_t) t * ld + c0 + tid] = sr[tid] + sr[64 + tid] + sr[128 + tid] + sr[192 + tid];
    }
}
// the same scores two query rows a block (rows 2 x, 2 x + 1): a K tile is staged once for both and each B fragment
// feeds four MMAs (rows' 64 heads x 2 = 8 row tiles); per pooled row and query row the head sums are formed exactly as
// rows_idx_score_tc_kernel forms them (4 groups of 16 heads, each in order, then the groups in order): the same scores
__global__ void __launch_bounds__(256) rows_idx_score_tc2_kernel(const float* q, const float* w, const uint16_t* k, int p0,
                                                                int T, float wscale, float* score, int ld) {
    using namespace nvcuda;
    extern __shared__ __align__(32) unsigned char smem_tc[];
    __half* sq = (__half*) smem_tc;                    // 128 x kTcLd (row 0's heads, then row 1's)
    __half* sk = sq + 128 * kTcLd;                     // 64 x kTcLd
    float* ss = (float*) (sk + 64 * kTcLd);            // 128 x kTcSld
    float* sw = ss + 128 * kTcSld;                     // 128 head weights
    float* sr = sw + 128;                              // 2 rows x 4 groups x 64 partial sums
    const int t0 = 2 * blockIdx.x, tid = threadIdx.x, wp = tid >> 5;
    const int nr = min(2, T - t0);
    const int nv0 = (p0 + t0 + 1) / 4, nv1 = nr > 1 ? (p0 + t0 + 2) / 4 : 0;
    const int n_vis = max(nv0, nv1), c_lo = blockIdx.y * kIdxRowsB;
    if (c_lo >= n_vis) return;
    const int c_hi = min(n_vis, c_lo + kIdxRowsB);
    for (int i = tid; i < 128 * kDsv4Ihd; i += 256) {
        const int r = i / (64 * kDsv4Ihd);
        const float v = r < nr ? q[(size_t) (t0 + r) * 64 * kDsv4Ihd + i % (64 * kDsv4Ihd)] : 0.0f;
        sq[(i / kDsv4Ihd) * kTcLd + i % kDsv4Ihd] = __float2half(v);
    }
    if (tid < 128) sw[tid] = (tid >> 6) < nr ? w[(size_t) (t0 + (tid >> 6)) * 64 + (tid & 63)] * wscale : 0.0f;
    const __half* kh = (const __half*) k;
    for (int c0 = c_lo; c0 < c_hi; c0 += 64) {
        __syncthreads();
        for (int i = tid; i < 64 * (kDsv4Ihd / 8); i += 256) {
            const int r = i / (kDsv4Ihd / 8), cc = (i % (kDsv4Ihd / 8)) * 8;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (c0 + r < c_hi) v = *(const uint4*) (kh + (size_t) (c0 + r) * kDsv4Ihd + cc);
            *(uint4*) (sk + r * kTcLd + cc) = v;
        }
        __syncthreads();
        {   // warp wp: key tile tj = wp & 3, row tiles ti = 4 (wp >> 2) .. + 3
            const int tj = wp & 3, ti0 = (wp >> 2) * 4;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) wmma::fill_fragment(acc[i], 0.0f);
#pragma unroll
            for (int kk = 0; kk < kDsv4Ihd; kk += 16) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
                wmma::load_matrix_sync(b, sk + tj * 16 * kTcLd + kk, kTcLd);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
                    wmma::load_matrix_sync(a, sq + (ti0 + i) * 16 * kTcLd + kk, kTcLd);
                    wmma::mma_sync(acc[i], a, b, acc[i]);
                }
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
                wmma::store_matrix_sync(ss + (ti0 + i) * 16 * kTcSld + tj * 16, acc[i], kTcSld, wmma::mem_row_major);
        }
        __syncthreads();
#pragma unroll
        for (int rep = 0; rep < 2; ++rep) {   // partial (row r, group g, pooled row c): heads 16 g .. 16 g + 15
            const int i = tid + rep * 256, r = i >> 8, g = (i >> 6) & 3, c = i & 63, h0 = g * 16;
            float v = 0.0f;
#pragma unroll
            for (int h = h0; h < h0 + 16; ++h) v += fmaxf(ss[(r * 64 + h) * kTcSld + c], 0.0f) * sw[r * 64 + h];
            sr[(r * 4 + g) * 64 + c] = v;
        }
        __syncthreads();
        if (tid < 128) {
            const int r = tid >> 6, c = tid & 63, nv = r ? nv1 : nv0;
            if (r < nr && c0 + c < min(nv, c_hi))
                score[(size_t) (t0 + r) * ld + c0 + c] = sr[(r * 4) * 64 + c] + sr[(r * 4 + 1) * 64 + c] +
                                                         sr[(r * 4 + 2) * 64 + c] + sr[(r * 4 + 3) * 64 + c];
        }
    }
}
// the top k of row t's visible scores (T blocks of 1024): all of them when there are k or fewer
__global__ void __launch_bounds__(1024) rows_topk_kernel(const float* score, int ld, int p0, int k, int* sel) {
    const int t = blockIdx.x, n = (p0 + t + 1) / 4;
    int* out = sel + (size_t) t * k;
    if (n <= k) {
        for (int i = threadIdx.x; i < n; i += blockDim.x) out[i] = i;
        return;
    }
    topk_select(score + (size_t) t * ld, n, k, out);
}
// the attention of row t's 64 heads (one block of 512 a row): q (raw, per-head norm + rope here) over the raw window
// (the ring below p0, raw16 from p0 on) and the visible pooled rows (sel: row t's k picks when more than k are visible,
// else all of them; HCA: all), online softmax with the sinks, the output's rope undone -> out16 (n_head x 512)
constexpr int kAttRows = 16, kQS = kDsv4Hd + 8;
__global__ void __launch_bounds__(512) rows_attn_kernel(const float* q, const float* sinks, const uint16_t* ring,
                                                       const uint16_t* raw16, const uint16_t* comp, int n_swa, int p0,
                                                       int ratio, const int* sel, int k, Dsv4Rope r, float eps,
                                                       uint16_t* out16) {
    extern __shared__ __align__(16) unsigned char smem[];
    __half* sq = (__half*) smem;                                   // 64 x kQS
    __half* srow = sq + 64 * kQS;                                  // kAttRows x kQS
    float* sp = (float*) (srow + kAttRows * kQS);                  // 64 x kAttRows
    float* sm = sp + 64 * kAttRows;                                // 64 running max
    float* sl = sm + 64;                                           // 64 running sum
    float* scr = sl + 64;                                          // 64 corrections
    const int t = blockIdx.x, tid = threadIdx.x, lane = tid & 31, wp = tid >> 5;
    const int pos = p0 + t;
    // q: per head (warp wp: heads wp, wp + 16, wp + 32, wp + 48) RMS norm, the tail's rope, FP16
    for (int h = wp; h < 64; h += 16) {
        const float* qh = q + ((size_t) t * 64 + h) * kDsv4Hd;
        float v[16];
        float ss = 0.0f;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            v[j] = qh[lane + 32 * j];
            ss += v[j] * v[j];
        }
        ss = wsum(ss);
        const float inv = rsqrtf(ss / (float) kDsv4Hd + eps);
#pragma unroll
        for (int j = 0; j < 16; ++j) v[j] *= inv;
#pragma unroll
        for (int j = 14; j < 16; ++j) {   // dims 448..511: pair (2 i, 2 i + 1) on lanes 2 i', 2 i' + 1
            const float o = __shfl_xor_sync(0xffffffffu, v[j], 1);
            float x0 = (lane & 1) ? o : v[j], x1 = (lane & 1) ? v[j] : o;
            rope_pair(x0, x1, (lane + 32 * (j - 14)) >> 1, (float) pos, r, false);
            v[j] = (lane & 1) ? x1 : x0;
        }
#pragma unroll
        for (int j = 0; j < 16; ++j) sq[h * kQS + lane + 32 * j] = __float2half_rn(v[j]);
    }
    if (tid < 64) {
        sm[tid] = sinks[tid];   // the sink starts the softmax: exp(sink - max) in the denominator
        sl[tid] = 1.0f;
    }
    const int n_raw = min(pos + 1, n_swa);
    const int n_vis = ratio > 0 ? (pos + 1) / ratio : 0;
    const int n_cmp = sel != nullptr ? min(n_vis, k) : n_vis;
    const int n_all = n_raw + n_cmp;
    const int hh = tid >> 3, dl = tid & 7;   // the output: head hh, dims dl + 8 j
    float acc[64];
#pragma unroll
    for (int j = 0; j < 64; ++j) acc[j] = 0.0f;
    const float scale = rsqrtf((float) kDsv4Hd);
    for (int i0 = 0; i0 < n_all; i0 += kAttRows) {
        __syncthreads();
        // the tile's rows into shared memory (64 uint4 a row)
        for (int i = tid; i < kAttRows * 64; i += 512) {
            const int rr = i >> 6, c = i & 63, ii = i0 + rr;
            uint4 v = make_uint4(0u, 0u, 0u, 0u);
            if (ii < n_all) {
                const uint16_t* src;
                if (ii < n_raw) {
                    const int pp = pos - n_raw + 1 + ii;
                    src = pp >= p0 ? raw16 + (size_t) (pp - p0) * kDsv4Hd : ring + (size_t) (pp % n_swa) * kDsv4Hd;
                } else {
                    const int cc = ii - n_raw;
                    src = comp + (size_t) (sel != nullptr && n_vis > k ? sel[(size_t) t * k + cc] : cc) * kDsv4Hd;
                }
                v = ((const uint4*) src)[c];
            }
            *(uint4*) (srow + rr * kQS + c * 8) = v;
        }
        __syncthreads();
        // the scores: thread (head hh, rows dl and dl + 8)
#pragma unroll
        for (int rq = 0; rq < 2; ++rq) {
            const int rr = dl + 8 * rq;
            const __half2* a2 = (const __half2*) (sq + hh * kQS);
            const __half2* b2 = (const __half2*) (srow + rr * kQS);
            float sv = 0.0f;
#pragma unroll 8
            for (int i = 0; i < kDsv4Hd / 2; ++i) {
                const float2 a = __half22float2(a2[i]), b = __half22float2(b2[i]);
                sv += a.x * b.x + a.y * b.y;
            }
            sp[hh * kAttRows + rr] = i0 + rr < n_all ? sv * scale : -INFINITY;
        }
        __syncthreads();
        // the online softmax per head (warp wp: heads 4 wp .. 4 wp + 3; lanes < kAttRows the rows)
        for (int h = 4 * wp; h < 4 * wp + 4; ++h) {
            const float v = lane < kAttRows ? sp[h * kAttRows + lane] : -INFINITY;
            const float mt = wmax(v);
            const float mo = sm[h], mn = fmaxf(mo, mt);
            const float p = lane < kAttRows && v > -INFINITY ? expf(v - mn) : 0.0f;
            const float ps = wsum(p);
            if (lane < kAttRows) sp[h * kAttRows + lane] = p;
            __syncwarp();
            if (lane == 0) {
                const float c = expf(mo - mn);
                scr[h] = c;
                sl[h] = sl[h] * c + ps;
                sm[h] = mn;
            }
        }
        __syncthreads();
        const float c = scr[hh];
#pragma unroll
        for (int j = 0; j < 64; ++j) acc[j] *= c;
        for (int rr = 0; rr < kAttRows; ++rr) {
            const float p = sp[hh * kAttRows + rr];
            if (p == 0.0f) continue;
            const __half* rv = srow + rr * kQS + dl;
#pragma unroll
            for (int j = 0; j < 64; ++j) acc[j] += p * __half2float(rv[8 * j]);
        }
    }
    __syncthreads();
    const float il = 1.0f / sl[hh];
#pragma unroll
    for (int j = 0; j < 64; ++j) acc[j] *= il;
    // the output's rope undone: dims 448..511 = j 56..63; pair (2 i, 2 i + 1): dl even / odd of the same head
#pragma unroll
    for (int j = 56; j < 64; ++j) {
        const float o = __shfl_xor_sync(0xffffffffu, acc[j], 1);
        float x0 = (dl & 1) ? o : acc[j], x1 = (dl & 1) ? acc[j] : o;
        const int d = dl + 8 * j;
        rope_pair(x0, x1, (d - (kDsv4Hd - kDsv4Rot)) >> 1, (float) pos, r, true);
        acc[j] = (dl & 1) ? x1 : x0;
    }
    uint16_t* o = out16 + ((size_t) t * 64 + hh) * kDsv4Hd;
#pragma unroll
    for (int j = 0; j < 64; ++j) o[dl + 8 * j] = f2h(acc[j]);
}
// ---- the same attention on tensor cores: a block (row t, 16 heads hg * 16 ..), 256 threads; tiles of 32 key rows
// (the shared K = V rows, FP16); pass 1 the exact softmax max / sum per head (the sink starts it), pass 2
// P = exp(s - m) (FP16) . V into wmma accumulators (no rescaling), then / sum, the output's rope undone.
constexpr int kTA = 32, kTAP = kTA + 8;
__device__ __forceinline__ const uint16_t* attn_row_src(int ii, int n_raw, int pos, int p0, int n_swa,
                                                        const uint16_t* raw16, const uint16_t* ring,
                                                        const uint16_t* comp, const int* sel, int t, int k, int n_vis) {
    if (ii < n_raw) {
        const int pp = pos - n_raw + 1 + ii;
        return pp >= p0 ? raw16 + (size_t) (pp - p0) * kDsv4Hd : ring + (size_t) (pp % n_swa) * kDsv4Hd;
    }
    const int cc = ii - n_raw;
    return comp + (size_t) (sel != nullptr && n_vis > k ? sel[(size_t) t * k + cc] : cc) * kDsv4Hd;
}
template <int HG>   // heads a block: 16 or 32 (fewer blocks load the same key rows)
__global__ void __launch_bounds__(256) rows_attn_tc_kernel(const float* q, const float* sinks, const uint16_t* ring,
                                                          const uint16_t* raw16, const uint16_t* comp, int n_swa, int p0,
                                                          int ratio, const int* sel, int k, Dsv4Rope r, float eps,
                                                          uint16_t* out16) {
    using namespace nvcuda;
    constexpr int RT = HG / 16, KP = 8 / (2 * RT), KW = kDsv4Hd / KP;   // row tiles; score k-parts; their width
    constexpr int CPT = HG * kTA / 256, TPH = kTA / CPT;               // softmax: columns a thread, threads a head
    constexpr int CT = 32 * RT / 8;                                     // PV: output tiles a warp
    extern __shared__ __align__(32) unsigned char smem[];
    __half* sq = (__half*) smem;                                   // HG x kQS
    __half* srow = sq + HG * kQS;                                  // kTA x kQS (the end: HG x 256 floats over it)
    float* spart = (float*) (srow + kTA * kQS);                    // KP x HG x kTA
    __half* sp16 = (__half*) (spart + KP * HG * kTA);              // HG x kTAP
    float* sm = (float*) (sp16 + HG * kTAP);                       // HG max
    float* sl = sm + HG;                                           // HG sums
    const int t = blockIdx.x, hg = blockIdx.y, tid = threadIdx.x, lane = tid & 31, wp = tid >> 5;
    const int pos = p0 + t;
    for (int hl = wp; hl < HG; hl += 8) {   // q: RMS norm, the tail's rope, FP16 (as the SIMT kernel)
        const int h = hg * HG + hl;
        const float* qh = q + ((size_t) t * 64 + h) * kDsv4Hd;
        float v[16];
        float ss = 0.0f;
#pragma unroll
        for (int jj = 0; jj < 16; ++jj) {
            v[jj] = qh[lane + 32 * jj];
            ss += v[jj] * v[jj];
        }
        ss = wsum(ss);
        const float inv = rsqrtf(ss / (float) kDsv4Hd + eps);
#pragma unroll
        for (int jj = 0; jj < 16; ++jj) v[jj] *= inv;
#pragma unroll
        for (int jj = 14; jj < 16; ++jj) {
            const float o = __shfl_xor_sync(0xffffffffu, v[jj], 1);
            float x0 = (lane & 1) ? o : v[jj], x1 = (lane & 1) ? v[jj] : o;
            rope_pair(x0, x1, (lane + 32 * (jj - 14)) >> 1, (float) pos, r, false);
            v[jj] = (lane & 1) ? x1 : x0;
        }
#pragma unroll
        for (int jj = 0; jj < 16; ++jj) sq[hl * kQS + lane + 32 * jj] = __float2half_rn(v[jj]);
    }
    if (tid < HG) {
        sm[tid] = sinks[hg * HG + tid];
        sl[tid] = 1.0f;
    }
    const int n_raw = min(pos + 1, n_swa);
    const int n_vis = ratio > 0 ? (pos + 1) / ratio : 0;
    const int n_cmp = sel != nullptr ? min(n_vis, k) : n_vis;
    const int n_all = n_raw + n_cmp;
    const float scale = rsqrtf((float) kDsv4Hd);
    // scores: warp (row tile st_r, key tile st_c, k-part kp)
    const int st_c = wp & 1, st_r = (wp >> 1) % RT, kp = wp / (2 * RT);
    const int hh = tid / TPH, c0 = (tid % TPH) * CPT;   // the softmax: head hh, key columns c0 ..
    // PV: warp wp the row tile pr, column tiles pc0 .. pc0 + CT - 1
    const int pr = wp % RT, pc0 = (wp / RT) * CT;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> oacc[CT];
#pragma unroll
    for (int i = 0; i < CT; ++i) wmma::fill_fragment(oacc[i], 0.0f);
    for (int pass = 0; pass < 2; ++pass) {
        for (int i0 = 0; i0 < n_all; i0 += kTA) {
            __syncthreads();
            for (int i = tid; i < kTA * 64; i += 256) {
                const int rr = i >> 6, c = i & 63, ii = i0 + rr;
                uint4 v = make_uint4(0u, 0u, 0u, 0u);
                if (ii < n_all)
                    v = ((const uint4*) attn_row_src(ii, n_raw, pos, p0, n_swa, raw16, ring, comp, sel, t, k, n_vis))[c];
                *(uint4*) (srow + rr * kQS + c * 8) = v;
            }
            __syncthreads();
            {
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> sacc;
                wmma::fill_fragment(sacc, 0.0f);
#pragma unroll
                for (int kk = 0; kk < KW; kk += 16) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
                    wmma::load_matrix_sync(a, sq + st_r * 16 * kQS + kp * KW + kk, kQS);
                    wmma::load_matrix_sync(b, srow + st_c * 16 * kQS + kp * KW + kk, kQS);
                    wmma::mma_sync(sacc, a, b, sacc);
                }
                wmma::store_matrix_sync(spart + kp * HG * kTA + st_r * 16 * kTA + st_c * 16, sacc, kTA,
                                        wmma::mem_row_major);
            }
            __syncthreads();
            float sv[CPT];
#pragma unroll
            for (int jj = 0; jj < CPT; ++jj) {
                const int c = c0 + jj;
                float v = 0.0f;
#pragma unroll
                for (int p = 0; p < KP; ++p) v += spart[p * HG * kTA + hh * kTA + c];
                sv[jj] = i0 + c < n_all ? v * scale : -INFINITY;
            }
            if (pass == 0) {   // the running max / sum of head hh (TPH lanes a head)
                float mt = -INFINITY;
#pragma unroll
                for (int jj = 0; jj < CPT; ++jj) mt = fmaxf(mt, sv[jj]);
#pragma unroll
                for (int o = TPH / 2; o > 0; o >>= 1) mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, o));
                const float mo = sm[hh], mn = fmaxf(mo, mt);
                float ps = 0.0f;
#pragma unroll
                for (int jj = 0; jj < CPT; ++jj) ps += sv[jj] > -INFINITY ? expf(sv[jj] - mn) : 0.0f;
#pragma unroll
                for (int o = TPH / 2; o > 0; o >>= 1) ps += __shfl_xor_sync(0xffffffffu, ps, o);
                __syncwarp();
                if (tid % TPH == 0) {
                    sl[hh] = sl[hh] * expf(mo - mn) + ps;
                    sm[hh] = mn;
                }
            } else {
                const float m = sm[hh];
#pragma unroll
                for (int jj = 0; jj < CPT; ++jj)
                    sp16[hh * kTAP + c0 + jj] = __float2half_rn(sv[jj] > -INFINITY ? expf(sv[jj] - m) : 0.0f);
                __syncthreads();
#pragma unroll
                for (int kk = 0; kk < kTA; kk += 16) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
                    wmma::load_matrix_sync(a, sp16 + pr * 16 * kTAP + kk, kTAP);
#pragma unroll
                    for (int i = 0; i < CT; ++i) {
                        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> bv;
                        wmma::load_matrix_sync(bv, srow + kk * kQS + (pc0 + i) * 16, kQS);
                        wmma::mma_sync(oacc[i], a, bv, oacc[i]);
                    }
                }
            }
        }
    }
    __syncthreads();
    // the end, in two halves of the 512 dims: HG x 256 floats over the key tile (33 KB, more than the 32 KB)
    float* so = (float*) srow;
    for (int half = 0; half < 2; ++half) {
        __syncthreads();
#pragma unroll
        for (int i = 0; i < CT; ++i) {
            const int col = (pc0 + i) * 16 - half * (kDsv4Hd / 2);
            if (col >= 0 && col < kDsv4Hd / 2)
                wmma::store_matrix_sync(so + pr * 16 * (kDsv4Hd / 2) + col, oacc[i], kDsv4Hd / 2, wmma::mem_row_major);
        }
        __syncthreads();
        for (int i = tid; i < HG * kDsv4Hd / 4; i += 256) {   // pairs (2 i, 2 i + 1): / sum, the tail's rope undone
            const int hl = i / (kDsv4Hd / 4), dl = 2 * (i % (kDsv4Hd / 4)), d = half * (kDsv4Hd / 2) + dl;
            const float il = 1.0f / sl[hl];
            float x0 = so[hl * (kDsv4Hd / 2) + dl] * il, x1 = so[hl * (kDsv4Hd / 2) + dl + 1] * il;
            if (d >= kDsv4Hd - kDsv4Rot) rope_pair(x0, x1, (d - (kDsv4Hd - kDsv4Rot)) >> 1, (float) pos, r, true);
            uint16_t* o = out16 + ((size_t) t * 64 + hg * HG + hl) * kDsv4Hd;
            o[d] = f2h(x0);
            o[d + 1] = f2h(x1);
        }
    }
}
// the sub-batch's last n_swa raw rows into the ring
__global__ void rows_ring_kernel(const uint16_t* raw16, uint16_t* ring, int n_swa, int p0, int T) {
    const int first = max(0, T - n_swa), t = first + blockIdx.x;
    if (t >= T) return;
    const int pos = p0 + t;
    for (int c = threadIdx.x; c < kDsv4Hd / 8; c += blockDim.x)
        ((uint4*) (ring + (size_t) (pos % n_swa) * kDsv4Hd))[c] = ((const uint4*) (raw16 + (size_t) t * kDsv4Hd))[c];
}

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) std::fprintf(stderr, "glm_dsv4 %s: %s\n", what, cudaGetErrorString(e));
}

}  // namespace

// (glm_fast.cu)
void quantize_q8_1(const float* x, void* xq, int n, cudaStream_t s);

void dsv4_kv_update(const Dsv4KvArgs& a, const Dsv4Layer& L, const Dsv4Rope& rope, int pos, cudaStream_t s) {
    const int blocks = L.ratio == 0 ? 1 : L.ratio == 4 ? 3 : 2;
    kv_update_kernel<<<blocks, 512, 0, s>>>(a, L, rope, pos);
    check("kv_update");
}

void dsv4_qr_norm(const float* qr, const float* w, float eps, int n, float* out, void* out_q8, cudaStream_t s) {
    qr_norm_kernel<<<1, 1024, 0, s>>>(qr, w, eps, n, out);
    quantize_q8_1(out, out_q8, n, s);
    check("qr_norm");
}

void dsv4_indexer(float* q, const float* w, int n_head, const Dsv4Layer& L, int n_comp, const Dsv4Rope& rope,
                  int pos, int top_k, float* score, int* sel, cudaStream_t s) {
    if (n_head != 64) {
        std::fprintf(stderr, "glm_dsv4 indexer: %d heads (64 only)\n", n_head);
        return;
    }
    idx_q_kernel<<<n_head, kDsv4Ihd, 0, s>>>(q, rope, pos);
    static const bool scalar = getenv("STRATA_DSV4_IDX_SCALAR") != nullptr;   // (the F32 kernel, for comparison)
    if (scalar)
        idx_score_kernel<<<(n_comp + kIdxRows - 1) / kIdxRows, 256, 0, s>>>(
            q, w, L.icomp, n_comp, 1.0f / std::sqrt((float) (kDsv4Ihd * n_head)), score);
    else
        idx_score_tc_kernel<<<(n_comp + kTcRows - 1) / kTcRows, 128, 0, s>>>(
            q, w, L.icomp, n_comp, 1.0f / std::sqrt((float) (kDsv4Ihd * n_head)), score);
    topk_kernel<<<1, 1024, 0, s>>>(score, n_comp, top_k, sel);
    check("indexer");
}

// Decode attention split over the keys (flash-decoding): every head reads the same latent rows (one 512-wide K = V
// head), so a block takes kSplitHeads heads and one chunk of rows - a row is read once for 8 heads, not once a
// head, and 64 / 8 x chunks blocks fill the card instead of 64.  Each block leaves its heads' unnormalised partial
// output, max and sum; attn_combine_kernel merges the chunks with the sink, normalises and undoes the tail's rope.
constexpr int kSplitHeads = 8, kSplitMax = 32;
__global__ void __launch_bounds__(256) attn_split_kernel(const float* q, const uint16_t* swa, const uint16_t* comp,
                                                         int n_swa, int pos, const int* sel, int n_sel, int rows,
                                                         Dsv4Rope r, float eps, float* part, float* ml) {
    constexpr int HB = kSplitHeads;
    extern __shared__ float smem_s[];
    float* sq = smem_s;                       // HB x 512
    float* sp = sq + HB * kDsv4Hd;            // HB x rows: scores, then exp(s - m)
    __shared__ const uint16_t* rowp[1024];
    __shared__ float sm[HB];
    const int h0 = blockIdx.x * HB, ch = blockIdx.y, t = threadIdx.x, lane = t & 31, wp = t >> 5;
    const int n_raw = min(pos + 1, n_swa), n_all = n_raw + n_sel;
    const int i0 = ch * rows, nr = max(0, min(rows, n_all - i0));
    for (int i = t; i < nr; i += 256) {
        const int ii = i0 + i;
        rowp[i] = ii < n_raw ? swa + (size_t) ((pos - n_raw + 1 + ii) % n_swa) * kDsv4Hd
                             : comp + (size_t) (sel != nullptr ? sel[ii - n_raw] : ii - n_raw) * kDsv4Hd;
    }
    {   // q of head h0 + wp: the RMS norm (no weight), the tail's rope
        const float* qh = q + (size_t) (h0 + wp) * kDsv4Hd;
        float v[16], ss = 0.0f;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            v[j] = qh[lane + 32 * j];
            ss += v[j] * v[j];
        }
        const float inv = rsqrtf(wsum(ss) / (float) kDsv4Hd + eps);
#pragma unroll
        for (int j = 0; j < 16; ++j) sq[wp * kDsv4Hd + lane + 32 * j] = v[j] * inv;
        __syncwarp();
        const int d = kDsv4Hd - kDsv4Rot + 2 * lane;   // 32 pairs, a lane each
        float x0 = sq[wp * kDsv4Hd + d], x1 = sq[wp * kDsv4Hd + d + 1];
        rope_pair(x0, x1, lane, (float) pos, r, false);
        sq[wp * kDsv4Hd + d] = x0;
        sq[wp * kDsv4Hd + d + 1] = x1;
    }
    __syncthreads();
    const float scale = rsqrtf((float) kDsv4Hd);
    // the scores: a warp a row, lane l dims 8 l .. 8 l + 7 and 256 + 8 l .. (two 16-byte loads a lane)
    for (int i = wp; i < nr; i += 8) {
        const uint4 ka = ((const uint4*) rowp[i])[lane], kb = ((const uint4*) rowp[i])[32 + lane];
        float k[16];
        const __half2* ha = (const __half2*) &ka;
        const __half2* hb = (const __half2*) &kb;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 fa = __half22float2(ha[j]), fb = __half22float2(hb[j]);
            k[2 * j] = fa.x;
            k[2 * j + 1] = fa.y;
            k[8 + 2 * j] = fb.x;
            k[8 + 2 * j + 1] = fb.y;
        }
#pragma unroll
        for (int h = 0; h < HB; ++h) {
            const float* qq = sq + h * kDsv4Hd;
            float s = 0.0f;
#pragma unroll
            for (int j = 0; j < 8; ++j) s += qq[8 * lane + j] * k[j] + qq[256 + 8 * lane + j] * k[8 + j];
            s = wsum(s);
            if (lane == 0) sp[h * rows + i] = s * scale;
        }
    }
    __syncthreads();
    {   // head wp: the chunk's max and sum, the weights in place
        float m = -INFINITY;
        for (int i = lane; i < nr; i += 32) m = fmaxf(m, sp[wp * rows + i]);
        m = wmax(m);
        float l = 0.0f;
        for (int i = lane; i < nr; i += 32) {
            const float e = expf(sp[wp * rows + i] - m);
            sp[wp * rows + i] = e;
            l += e;
        }
        l = wsum(l);
        if (lane == 0) {
            sm[wp] = m;
            ml[((size_t) ch * 64 + h0 + wp) * 2] = m;
            ml[((size_t) ch * 64 + h0 + wp) * 2 + 1] = l;
        }
    }
    __syncthreads();
    // the partial output: thread t dims 2 t, 2 t + 1 for the 8 heads
    float acc[HB][2];
#pragma unroll
    for (int h = 0; h < HB; ++h) acc[h][0] = acc[h][1] = 0.0f;
    for (int i = 0; i < nr; ++i) {
        const float2 v = __half22float2(((const __half2*) rowp[i])[t]);
#pragma unroll
        for (int h = 0; h < HB; ++h) {
            const float w = sp[h * rows + i];
            acc[h][0] += w * v.x;
            acc[h][1] += w * v.y;
        }
    }
#pragma unroll
    for (int h = 0; h < HB; ++h)
        ((float2*) (part + ((size_t) ch * 64 + h0 + h) * kDsv4Hd))[t] = make_float2(acc[h][0], acc[h][1]);
}

__global__ void __launch_bounds__(256) attn_combine_kernel(const float* sinks, const float* part, const float* ml,
                                                           int chunks, int pos, Dsv4Rope r, float* out) {
    __shared__ float so[kDsv4Hd];
    const int h = blockIdx.x, t = threadIdx.x;
    float M = sinks[h];
    for (int c = 0; c < chunks; ++c) M = fmaxf(M, ml[((size_t) c * 64 + h) * 2]);
    float den = expf(sinks[h] - M), o0 = 0.0f, o1 = 0.0f;
    for (int c = 0; c < chunks; ++c) {
        const float m = ml[((size_t) c * 64 + h) * 2], l = ml[((size_t) c * 64 + h) * 2 + 1];
        if (l == 0.0f) continue;
        const float f = expf(m - M);
        den += l * f;
        const float* pp = part + ((size_t) c * 64 + h) * kDsv4Hd;
        o0 += pp[t] * f;
        o1 += pp[t + 256] * f;
    }
    const float id = 1.0f / den;
    so[t] = o0 * id;
    so[t + 256] = o1 * id;
    __syncthreads();
    if (t < kDsv4Rot / 2) {   // the output's rope undone (the tail)
        const int d = kDsv4Hd - kDsv4Rot + 2 * t;
        float x0 = so[d], x1 = so[d + 1];
        rope_pair(x0, x1, t, (float) pos, r, true);
        so[d] = x0;
        so[d + 1] = x1;
    }
    __syncthreads();
    out[(size_t) h * kDsv4Hd + t] = so[t];
    out[(size_t) h * kDsv4Hd + t + 256] = so[t + 256];
}

void dsv4_attn(const float* q, const float* sinks, const Dsv4Layer& L, int n_swa, int pos, const int* sel, int n_sel,
               const Dsv4Rope& rope, float eps, int n_head, float* out, cudaStream_t s) {
    const int n_all = std::min(pos + 1, n_swa) + n_sel;
    // the split kernel (default; STRATA_DSV4_ATTN_SPLIT=0 = one block a head): ~32 rows a chunk, at most 32 chunks
    static const bool split = !(getenv("STRATA_DSV4_ATTN_SPLIT") && getenv("STRATA_DSV4_ATTN_SPLIT")[0] == '0');
    const int chunks = std::min(kSplitMax, std::max(1, (n_all + 31) / 32)), rows = (n_all + chunks - 1) / chunks;
    if (split && n_head == 64 && n_all > 0 && rows <= 1024) {
        static float* scratch[16] = {};
        int dev = 0;
        cudaGetDevice(&dev);
        if (scratch[dev] == nullptr &&
            cudaMalloc(&scratch[dev], (size_t) kSplitMax * 64 * (kDsv4Hd + 2) * sizeof(float)) != cudaSuccess) {
            std::fprintf(stderr, "glm_dsv4 attention: no split scratch\n");
            return;
        }
        float* part = scratch[dev];
        float* ml = part + (size_t) kSplitMax * 64 * kDsv4Hd;
        const size_t sm = (size_t) kSplitHeads * (kDsv4Hd + rows) * sizeof(float);
        static size_t opted_s = 48 * 1024;
        if (sm > opted_s) {
            cudaFuncSetAttribute(attn_split_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sm);
            opted_s = sm;
        }
        attn_split_kernel<<<dim3(64 / kSplitHeads, chunks), 256, sm, s>>>(q, L.swa, L.comp, n_swa, pos, sel, n_sel, rows,
                                                                          rope, eps, part, ml);
        attn_combine_kernel<<<64, 256, 0, s>>>(sinks, part, ml, chunks, pos, rope, out);
        check("attn_split");
        return;
    }
    const size_t smem = (size_t) (((n_all + 31) & ~31) + 8 * kDsv4Hd) * sizeof(float);   // scores + the warps' sums
    static size_t opted = 48 * 1024;
    if (smem > opted) {
        cudaFuncSetAttribute(attn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) std::min<size_t>(smem, 90 * 1024));
        opted = std::min<size_t>(smem, 90 * 1024);
    }
    attn_kernel<<<n_head, 256, smem, s>>>(q, sinks, L.swa, L.comp, n_swa, pos, sel, n_sel, rope, eps, out);
    check("attn");
}

void dsv4_hc_head(const float* R, const uint16_t* fn, const float* scale, const float* base, float hc_eps,
                  const float* norm_w, float eps, int n_embd, float* x, void* xq, cudaStream_t s) {
    hc_head_kernel<<<1, 1024, 0, s>>>(R, fn, scale, base, hc_eps, norm_w, eps, n_embd, x);
    quantize_q8_1(x, xq, n_embd, s);
    check("hc_head");
}

// ---- the prompt path
void dsv4_rows_qr(const float* qr_raw, const float* w, float eps, int T, int n, float* qr, uint16_t* qr16,
                  cudaStream_t s) {
    if (T <= 0) return;
    rows_qr_kernel<<<T, 1024, 0, s>>>(qr_raw, w, eps, n, qr, qr16);
    check("rows_qr");
}
void dsv4_rows_kv(const float* kv_raw, const float* w, float eps, int T, int p0, const Dsv4Rope& rope, uint16_t* raw16,
                  cudaStream_t s) {
    if (T <= 0) return;
    rows_kv_kernel<<<T, 512, 0, s>>>(kv_raw, w, eps, p0, rope, raw16);
    check("rows_kv");
}
void dsv4_rows_comp(const float* kv, const float* sc, int T, int p0, const float* ape, const float* norm, float eps,
                    const Dsv4Layer& L, const Dsv4Rope& rope, bool indexer, cudaStream_t s) {
    if (T <= 0 || L.ratio == 0) return;
    static const bool serial = getenv("STRATA_DSV4_COMP_SERIAL") != nullptr;   // (the position-after-position form)
    if (serial) {
        if (indexer)
            rows_comp_kernel<<<1, 512, 0, s>>>(kv, sc, T, p0, ape, norm, eps, L.ist_kv, L.ist_sc, 4, true, L.icomp, rope);
        else
            rows_comp_kernel<<<1, 512, 0, s>>>(kv, sc, T, p0, ape, norm, eps, L.st_kv, L.st_sc, L.ratio, false, L.comp, rope);
        check("rows_comp");
        return;
    }
    const int ratio = indexer ? 4 : L.ratio, hd = indexer ? kDsv4Ihd : kDsv4Hd, W = ratio == 4 ? 2 * hd : hd;
    float* st_kv = indexer ? L.ist_kv : L.st_kv;
    float* st_sc = indexer ? L.ist_sc : L.st_sc;
    const CompSrc S{kv, sc, ape, st_kv, st_sc, W, ratio, p0, p0 - p0 % ratio};
    const int e = p0 + T, first_at = p0 / ratio, n_rows = e / ratio - first_at;   // pooled rows ending in [p0, e)
    if (n_rows > 0)
        rows_comp_par_kernel<<<n_rows, 512, 0, s>>>(S, first_at, norm, eps, indexer, indexer ? L.icomp : L.comp, rope);
    comp_state_kernel<<<1, 512, 0, s>>>(S, e, st_kv, st_sc);
    check("rows_comp_par");
}
void dsv4_rows_indexer(float* q, const float* w, int n_head, const Dsv4Layer& L, int T, int p0, int top_k,
                       const Dsv4Rope& rope, float* score, int ld, int* sel, cudaStream_t s) {
    if (T <= 0) return;
    if (n_head != 64) {
        std::fprintf(stderr, "glm_dsv4 indexer: %d heads (64 only)\n", n_head);
        return;
    }
    const int max_vis = (p0 + T) / 4;
    // STRATA_DSV4_IDX_TIME=1: the q / score / top-k kernels' GPU time, summed over the calls (printed every 43 calls)
    static const bool itime = getenv("STRATA_DSV4_IDX_TIME") != nullptr;
    static cudaEvent_t ie[4];
    static double ims[3] = {0, 0, 0};
    static long icalls = 0;
    if (itime && icalls == 0)
        for (auto& e : ie) cudaEventCreate(&e);
    if (itime) cudaEventRecord(ie[0], s);
    rows_idx_q_kernel<<<T * n_head, kDsv4Ihd, 0, s>>>(q, n_head, p0, rope);
    if (itime) cudaEventRecord(ie[1], s);
    static const bool scalar = getenv("STRATA_DSV4_IDX_SCALAR") != nullptr;   // (the F32 kernel, for comparison)
    if (max_vis > top_k && scalar) {
        rows_idx_score_kernel<<<dim3(T, (max_vis + kIdxRowsB - 1) / kIdxRowsB), 256, 0, s>>>(
            q, w, L.icomp, p0, 1.0f / std::sqrt((float) (kDsv4Ihd * n_head)), score, ld);
    } else if (max_vis > top_k && !getenv("STRATA_DSV4_IDX_TC1")) {   // (STRATA_DSV4_IDX_TC1: one query row a block)
        const size_t smem = (size_t) (128 + 64) * kTcLd * 2 + (size_t) (128 * kTcSld + 128 + 512) * 4;
        static bool opted2 = false;
        if (!opted2) {
            cudaFuncSetAttribute(rows_idx_score_tc2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem);
            opted2 = true;
        }
        rows_idx_score_tc2_kernel<<<dim3((T + 1) / 2, (max_vis + kIdxRowsB - 1) / kIdxRowsB), 256, smem, s>>>(
            q, w, L.icomp, p0, T, 1.0f / std::sqrt((float) (kDsv4Ihd * n_head)), score, ld);
    } else if (max_vis > top_k) {
        const size_t smem = (size_t) 2 * 64 * kTcLd * 2 + (size_t) (64 * kTcSld + 64 + 256) * 4;
        static bool opted = false;
        if (!opted) {
            cudaFuncSetAttribute(rows_idx_score_tc_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem);
            opted = true;
        }
        rows_idx_score_tc_kernel<<<dim3(T, (max_vis + kIdxRowsB - 1) / kIdxRowsB), 256, smem, s>>>(
            q, w, L.icomp, p0, 1.0f / std::sqrt((float) (kDsv4Ihd * n_head)), score, ld);
    }
    if (itime) cudaEventRecord(ie[2], s);
    rows_topk_kernel<<<T, 1024, 0, s>>>(score, ld, p0, top_k, sel);
    check("rows_indexer");
    if (itime) {
        cudaEventRecord(ie[3], s);
        cudaEventSynchronize(ie[3]);
        for (int i = 0; i < 3; ++i) { float m = 0; cudaEventElapsedTime(&m, ie[i], ie[i + 1]); ims[i] += m; }
        if (++icalls % 43 == 0)
            std::fprintf(stderr, "dsv4 indexer prof (%ld calls): q %.0f ms, score %.0f ms, top-k %.0f ms\n", icalls,
                         ims[0], ims[1], ims[2]);
    }
}
void dsv4_rows_attn(const float* q, const float* sinks, const Dsv4Layer& L, const uint16_t* raw16, int n_swa, int p0,
                    int T, const int* sel, int k, const Dsv4Rope& rope, float eps, int n_head, uint16_t* out16,
                    cudaStream_t s) {
    if (T <= 0) return;
    if (n_head != 64) {
        std::fprintf(stderr, "glm_dsv4 attention: %d heads (64 only)\n", n_head);
        return;
    }
    static const bool simt = getenv("STRATA_DSV4_ATTN_SIMT") != nullptr;   // (the SIMT kernel, for comparison)
    if (!simt) {
        static const int hg = getenv("STRATA_DSV4_ATTN_HG") ? std::atoi(getenv("STRATA_DSV4_ATTN_HG")) : 32;
        auto bytes = [](int HG) {
            return (size_t) (HG + kTA) * kQS * 2 + (size_t) (8 / (2 * (HG / 16))) * HG * kTA * 4 +
                   (size_t) HG * kTAP * 2 + (size_t) 2 * HG * 4;
        };
        const size_t smem = bytes(hg);
        if (hg == 16) {
            static bool o16 = false;
            if (!o16) { cudaFuncSetAttribute(rows_attn_tc_kernel<16>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem); o16 = true; }
            rows_attn_tc_kernel<16><<<dim3(T, 4), 256, smem, s>>>(q, sinks, L.swa, raw16, L.comp, n_swa, p0, L.ratio, sel,
                                                                   k, rope, eps, out16);
        } else {
            static bool o32 = false;
            if (!o32) { cudaFuncSetAttribute(rows_attn_tc_kernel<32>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem); o32 = true; }
            rows_attn_tc_kernel<32><<<dim3(T, 2), 256, smem, s>>>(q, sinks, L.swa, raw16, L.comp, n_swa, p0, L.ratio, sel,
                                                                   k, rope, eps, out16);
        }
        check("rows_attn_tc");
        return;
    }
    const size_t smem = (size_t) (64 + kAttRows) * kQS * 2 + (size_t) (64 * kAttRows + 3 * 64) * 4;
    static bool opted = false;
    if (!opted) {
        cudaFuncSetAttribute(rows_attn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem);
        opted = true;
    }
    rows_attn_kernel<<<T, 512, smem, s>>>(q, sinks, L.swa, raw16, L.comp, n_swa, p0, L.ratio, sel, k, rope, eps, out16);
    check("rows_attn");
}
void dsv4_rows_ring(const uint16_t* raw16, const Dsv4Layer& L, int n_swa, int p0, int T, cudaStream_t s) {
    if (T <= 0) return;
    rows_ring_kernel<<<std::min(T, n_swa), 64, 0, s>>>(raw16, L.swa, n_swa, p0, T);
    check("rows_ring");
}

namespace {
__global__ void mtp_input_kernel(const float* ep, const float* hp, float* R, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 4 * n) R[i] = ep[i % n] + hp[i];
}
__global__ void mtp_combine_kernel(const float* y, Dsv4MtpW w, int k, const float* sh, float* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float a = sh ? sh[i] : 0.0f;
    for (int j = 0; j < k; ++j) a += w.w[j] * y[(size_t) j * n + i];
    out[i] = a;
}
}  // namespace
void dsv4_mtp_input(const float* ep, const float* hp, float* R, int n, cudaStream_t s) {
    mtp_input_kernel<<<(4 * n + 255) / 256, 256, 0, s>>>(ep, hp, R, n);
    check("mtp_input");
}
void dsv4_mtp_combine(const float* y, Dsv4MtpW w, int k, const float* sh, float* out, int n, cudaStream_t s) {
    mtp_combine_kernel<<<(n + 255) / 256, 256, 0, s>>>(y, w, k, sh, out, n);
    check("mtp_combine");
}

}  // namespace strata::kernels::glmf
