// src/kernels/cuda/glm_dsv4.cu - DeepSeek-V4 Flash's attention for the Glm5Model fast path (one token): the cache
// update (raw row + compressors), the indexer's pick, the attention with sinks, the mHC head.  See glm_dsv4.hpp; the
// arithmetic follows antirez's ds4.c (the CPU reference) step by step.
#include "strata/kernels/glm_dsv4.hpp"

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>

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

// block 0: the raw row; block 1: the attention compressor (ratio > 0); block 2: the indexer compressor (ratio 4).
// 512 threads.
__global__ void __launch_bounds__(512) kv_update_kernel(const Dsv4KvArgs a, const Dsv4Layer L, const Dsv4Rope r,
                                                        int pos) {
    __shared__ float buf[kDsv4Hd];
    __shared__ float red[32];
    const int d = threadIdx.x;
    if (blockIdx.x == 0) {
        float v = a.kv_raw[d];
        const float ss = bsum(v * v, red);
        v = v * rsqrtf(ss / (float) kDsv4Hd + a.eps) * a.kv_norm[d];
        v = rope_tail512(v, buf, (float) pos, r, false);
        v = fp8_448(v, red);
        L.swa[(size_t) (pos % a.n_swa) * kDsv4Hd + d] = f2h(v);
        return;
    }
    const int ratio = L.ratio;
    const bool emit = (pos + 1) % ratio == 0;
    const int idx = (pos + 1) / ratio - 1;
    if (blockIdx.x == 1) {
        const int W = ratio == 4 ? 2 * kDsv4Hd : kDsv4Hd;
        comp_store(L.st_kv, L.st_sc, a.comp_kv, a.comp_sc, a.comp_ape, W, ratio, pos);
        __syncthreads();
        if (!emit) return;
        float v = pool_dim(L.st_kv, L.st_sc, d, kDsv4Hd, ratio);
        __syncthreads();
        if (ratio == 4) comp_shift4(L.st_kv, L.st_sc, W);
        const float ss = bsum(v * v, red);
        v = v * rsqrtf(ss / (float) kDsv4Hd + a.eps) * a.comp_norm[d];
        v = rope_tail512(v, buf, (float) (pos + 1 - ratio), r, false);
        v = fp8_448(v, red);
        L.comp[(size_t) idx * kDsv4Hd + d] = f2h(v);
        return;
    }
    // block 2: the indexer compressor (head_dim 128, window width 256), then Hadamard and FP4
    const int W = 2 * kDsv4Ihd;
    comp_store(L.ist_kv, L.ist_sc, a.icomp_kv, a.icomp_sc, a.icomp_ape, W, 4, pos);
    __syncthreads();
    if (!emit) return;
    const bool on = d < kDsv4Ihd;
    float v = on ? pool_dim(L.ist_kv, L.ist_sc, d, kDsv4Ihd, 4) : 0.0f;
    __syncthreads();
    comp_shift4(L.ist_kv, L.ist_sc, W);
    const float ss = bsum(v * v, red);
    if (on) buf[d] = v * rsqrtf(ss / (float) kDsv4Ihd + a.eps) * a.icomp_norm[d];
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
        L.icomp[(size_t) idx * kDsv4Ihd + d] = f2h(x);
    }
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

// the top k of n scores (any order): a radix select on the order-preserving keys (3 passes of 11 / 11 / 10 bits), then
// every key above the threshold and the lowest-indexed equal ones.  One block of 1024 threads.
__device__ __forceinline__ uint32_t okey(float f) {
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);   // larger float -> larger key
}
__global__ void __launch_bounds__(1024) topk_kernel(const float* score, int n, int k, int* sel) {
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
        if (t == 0) {   // from the top bin down: the bin where the count reaches s_need
            int need = s_need, b = nb - 1;
            for (; b > 0; --b) {
                if ((int) hist[b] >= need) break;
                need -= (int) hist[b];
            }
            s_bin = b;
            s_need = need;
            s_prefix = pre | ((uint32_t) b << sh);
            s_mask = msk | ((uint32_t) (nb - 1) << sh);
        }
        __syncthreads();
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
    // the equal ones in index order: one thread (they are few unless the scores tie at large scale)
    if (t == 0) {
        int o = s_out, need = s_need;
        for (int i = 0; i < n && need > 0; ++i)
            if (okey(score[i]) == thr) {
                sel[o++] = i;
                --need;
            }
    }
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
    // the output: thread t dims t and t + 256
    float o0 = 0.0f, o1 = 0.0f;
    for (int i = 0; i < n_all; ++i) {
        const uint16_t* kr = row(i);
        const float w = sc[i];
        o0 += w * h2f(kr[t]);
        o1 += w * h2f(kr[t + 256]);
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
    idx_score_kernel<<<(n_comp + kIdxRows - 1) / kIdxRows, 256, 0, s>>>(
        q, w, L.icomp, n_comp, 1.0f / std::sqrt((float) (kDsv4Ihd * n_head)), score);
    topk_kernel<<<1, 1024, 0, s>>>(score, n_comp, top_k, sel);
    check("indexer");
}

void dsv4_attn(const float* q, const float* sinks, const Dsv4Layer& L, int n_swa, int pos, const int* sel, int n_sel,
               const Dsv4Rope& rope, float eps, int n_head, float* out, cudaStream_t s) {
    const int n_all = std::min(pos + 1, n_swa) + n_sel;
    const size_t smem = (size_t) n_all * sizeof(float);
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

}  // namespace strata::kernels::glmf
