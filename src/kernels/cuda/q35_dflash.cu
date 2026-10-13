// src/kernels/cuda/q35_dflash.cu - see include/strata/kernels/q35_dflash.hpp.
#include "strata/kernels/q35_dflash.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

constexpr int kHd = 128;          // the drafter's head size
constexpr int kTopKMax = 32;
constexpr int kSlices = 64;       // the top-k's first step: slices of a row

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ float block_sum(float v, float* red) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = (int) (blockDim.x >> 5);
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) red[w] = v;
    __syncthreads();
    float t = 0.0f;
    for (int i = 0; i < nw; ++i) t += red[i];
    return t;
}

// block (head, row), 64 threads: rms over the head * w, NeoX rope of the 128 dims at pos0 + row
__global__ void head_norm_rope_kernel(float* __restrict__ v, int n_heads, const float* __restrict__ w, float eps,
                                      int64_t pos0, float theta_base) {
    __shared__ float red[32];
    const int h = blockIdx.x, row = blockIdx.y, j = threadIdx.x;
    float* p = v + ((size_t) row * n_heads + h) * kHd;
    const float x0 = p[j], x1 = p[j + kHd / 2];
    const float r = rsqrtf(block_sum(x0 * x0 + x1 * x1, red) / (float) kHd + eps);
    const float a = x0 * r * w[j], b = x1 * r * w[j + kHd / 2];
    const float theta = (float) (pos0 + row) * powf(theta_base, -2.0f * (float) j / (float) kHd);
    float sn, cs;
    sincosf(theta, &sn, &cs);
    p[j] = a * cs - b * sn;
    p[j + kHd / 2] = a * sn + b * cs;
}

__global__ void conv_kernel(const float* __restrict__ x, const float* __restrict__ c, int ldc,
                            const float* __restrict__ base, int side, int E, int taps, int group,
                            float* __restrict__ out) {
    const int t = blockIdx.x, ng = E / group;
    const float* ct = c + (size_t) t * ldc + (size_t) side * taps * ng;
    const float* bs = base + (size_t) side * taps * E;
    for (int e = threadIdx.x; e < E; e += blockDim.x) {
        float acc = 0.0f;
        for (int tap = 0; tap < taps && tap <= t; ++tap)
            acc += (ct[tap * ng + e / group] + bs[(size_t) tap * E + e]) * x[(size_t) (t - tap) * E + e];
        out[(size_t) t * E + e] = acc;
    }
}

__global__ void ring_store_kernel(const float* __restrict__ K, const float* __restrict__ V, int kvd, int64_t pos0,
                                  int ring, __half* __restrict__ Kr, __half* __restrict__ Vr, int64_t* __restrict__ rp) {
    const int r = blockIdx.x;
    const size_t slot = (size_t) ((pos0 + r) % ring);
    for (int i = threadIdx.x; i < kvd; i += blockDim.x) {
        Kr[slot * kvd + i] = __float2half(K[(size_t) r * kvd + i]);
        Vr[slot * kvd + i] = __float2half(V[(size_t) r * kvd + i]);
    }
    if (threadIdx.x == 0) rp[slot] = pos0 + r;
}

// block (head, row), kHd threads: see the header (upstream glm_dflash.cu's attn_kernel)
__global__ void attn_kernel(const float* __restrict__ Q, const __half* __restrict__ Kr, const __half* __restrict__ Vr,
                            const int64_t* __restrict__ rp, const float* __restrict__ Kb, const float* __restrict__ Vb,
                            int rows, int n_head, int n_kv, int64_t n_past, int ring, int window, float scale,
                            float* __restrict__ out) {
    extern __shared__ float sc[];
    __shared__ float qs[kHd];
    __shared__ float red[32];
    const int h = blockIdx.x, i = blockIdx.y, d = threadIdx.x, lane = d & 31, wid = d >> 5, nw = kHd / 32;
    const int kvh = h / (n_head / n_kv), kvd = n_kv * kHd;
    const int64_t P = n_past + i;
    const int64_t lo = P - window + 1 > 0 ? P - window + 1 : 0;
    const int nc = n_past > lo ? (int) (n_past - lo) : 0, nk = nc + rows;
    qs[d] = Q[((size_t) i * n_head + h) * kHd + d];
    __syncthreads();
    for (int k = wid; k < nk; k += nw) {
        float acc = 0.0f;
        bool ok = true;
        if (k < nc) {
            const int64_t p = lo + k;
            const size_t slot = (size_t) (p % ring);
            ok = rp[slot] == p;
            const __half2* kr = (const __half2*) (Kr + slot * kvd + (size_t) kvh * kHd);
            const float2 kv = __half22float2(kr[lane]), kv2 = __half22float2(kr[lane + 32]);
            acc = qs[2 * lane] * kv.x + qs[2 * lane + 1] * kv.y + qs[64 + 2 * lane] * kv2.x + qs[64 + 2 * lane + 1] * kv2.y;
        } else {
            const float* kb = Kb + (size_t) (k - nc) * kvd + (size_t) kvh * kHd;
            for (int e = lane; e < kHd; e += 32) acc += qs[e] * kb[e];
        }
        acc = warp_sum(acc);
        if (lane == 0) sc[k] = ok ? acc * scale : -INFINITY;
    }
    __syncthreads();
    float m = -INFINITY;
    for (int k = d; k < nk; k += kHd) m = fmaxf(m, sc[k]);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if (lane == 0) red[wid] = m;
    __syncthreads();
    m = red[0];
    for (int w = 1; w < nw; ++w) m = fmaxf(m, red[w]);
    __syncthreads();
    float sum = 0.0f;
    for (int k = d; k < nk; k += kHd) {
        const float e = sc[k] == -INFINITY ? 0.0f : __expf(sc[k] - m);
        sc[k] = e;
        sum += e;
    }
    sum = block_sum(sum, red);
    __syncthreads();
    float acc = 0.0f;
    for (int k = 0; k < nk; ++k) {
        const float pk = sc[k];
        if (pk == 0.0f) continue;
        if (k < nc) {
            const size_t slot = (size_t) ((lo + k) % ring);
            acc += pk * __half2float(Vr[slot * kvd + (size_t) kvh * kHd + d]);
        } else {
            acc += pk * Vb[(size_t) (k - nc) * kvd + (size_t) kvh * kHd + d];
        }
    }
    out[((size_t) i * n_head + h) * kHd + d] = sum > 0.0f ? acc / sum : 0.0f;
}

__device__ __forceinline__ bool tk_before(float v, int i, float bv, int bi) { return v > bv || (v == bv && i < bi); }
__device__ __forceinline__ bool tk_after(float v, int i, float lv, int li) { return v < lv || (v == lv && i > li); }

__global__ void __launch_bounds__(256) topk_part_kernel(const float* __restrict__ x, int64_t ld, int n, int k,
                                                        int slice, int* __restrict__ part_i, float* __restrict__ part_v) {
    extern __shared__ float xs[];
    __shared__ float bv[8];
    __shared__ int bi[8];
    __shared__ float last_v;
    __shared__ int last_i;
    const int lo = blockIdx.x * slice, len = max(0, min(n - lo, slice));
    const float* xr = x + (size_t) blockIdx.y * ld + lo;
    for (int i = threadIdx.x; i < len; i += blockDim.x) xs[i] = xr[i];
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = (int) (blockDim.x >> 5);
    if (threadIdx.x == 0) { last_v = INFINITY; last_i = -1; }
    __syncthreads();
    const size_t o = ((size_t) blockIdx.y * gridDim.x + blockIdx.x) * k;
    for (int r = 0; r < k; ++r) {
        const float lv = last_v;
        const int li = last_i;
        float v = -INFINITY;
        int vi = 0x7fffffff;
        for (int i = threadIdx.x; i < len; i += blockDim.x) {
            const float y = xs[i];
            if (tk_after(y, lo + i, lv, li) && tk_before(y, lo + i, v, vi)) { v = y; vi = lo + i; }
        }
#pragma unroll
        for (int d = 16; d > 0; d >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, v, d);
            const int oi = __shfl_xor_sync(0xffffffffu, vi, d);
            if (tk_before(ov, oi, v, vi)) { v = ov; vi = oi; }
        }
        if (lane == 0) { bv[w] = v; bi[w] = vi; }
        __syncthreads();
        if (threadIdx.x == 0) {
            float b = bv[0];
            int bj = bi[0];
            for (int j = 1; j < nw; ++j)
                if (tk_before(bv[j], bi[j], b, bj)) { b = bv[j]; bj = bi[j]; }
            part_i[o + r] = bj;
            part_v[o + r] = b;
            last_v = b;
            last_i = bj;
        }
        __syncthreads();
    }
}

__global__ void __launch_bounds__(32) topk_merge_kernel(const int* __restrict__ part_i, const float* __restrict__ part_v,
                                                        int k, int* __restrict__ ids, float* __restrict__ vals) {
    extern __shared__ float cv[];
    const int c = kSlices * k;
    int* ci = (int*) (cv + c);
    const size_t o = (size_t) blockIdx.x * c;
    for (int j = threadIdx.x; j < c; j += 32) { cv[j] = part_v[o + j]; ci[j] = part_i[o + j]; }
    __syncwarp();
    float lv = INFINITY;
    int li = -1;
    for (int r = 0; r < k; ++r) {
        float v = -INFINITY;
        int vi = 0x7fffffff;
        for (int j = threadIdx.x; j < c; j += 32) {
            const float y = cv[j];
            const int yi = ci[j];
            if (tk_after(y, yi, lv, li) && tk_before(y, yi, v, vi)) { v = y; vi = yi; }
        }
#pragma unroll
        for (int d = 16; d > 0; d >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, v, d);
            const int oi = __shfl_xor_sync(0xffffffffu, vi, d);
            if (tk_before(ov, oi, v, vi)) { v = ov; vi = oi; }
        }
        if (threadIdx.x == 0) { ids[(size_t) blockIdx.x * k + r] = vi; vals[(size_t) blockIdx.x * k + r] = v; }
        lv = v;
        li = vi;
    }
}

// `rank` threads: the greedy walk (upstream's walk_kernel with FP16 tables)
__global__ void walk_kernel(const int* __restrict__ cand, const float* __restrict__ lg, const float* __restrict__ gate,
                            int k, int rank, const __half* __restrict__ pt, const __half* __restrict__ st, int anchor,
                            int n, int* __restrict__ tok, float* __restrict__ conf) {
    __shared__ float as[1024];
    __shared__ float s[kTopKMax];
    __shared__ int pred;
    const int r = threadIdx.x, lane = r & 31, w = r >> 5, nw = (int) (blockDim.x >> 5);
    if (r == 0) pred = anchor;
    __syncthreads();
    for (int i = 0; i < n; ++i) {
        as[r] = __half2float(pt[(size_t) pred * rank + r]) * gate[(size_t) i * rank + r];
        __syncthreads();
        for (int j = w; j < k; j += nw) {
            const int c = cand[(size_t) i * k + j];
            float t = 0.0f;
            for (int e = lane; e < rank; e += 32) t += as[e] * __half2float(st[(size_t) c * rank + e]);
            t = warp_sum(t);
            if (lane == 0) s[j] = lg[(size_t) i * k + j] + t;
        }
        __syncthreads();
        if (r == 0) {
            int b = 0;
            for (int j = 1; j < k; ++j)
                if (s[j] > s[b]) b = j;
            float z = 0.0f;
            for (int j = 0; j < k; ++j) z += __expf(s[j] - s[b]);
            tok[i] = cand[(size_t) i * k + b];
            conf[i] = z > 0.0f ? 1.0f / z : 0.0f;
            pred = tok[i];
        }
        __syncthreads();
    }
}

}  // namespace

void q35_df_head_norm_rope(float* x, int rows, int n_heads, const float* w, float eps, int64_t pos0, float theta,
                           void* stream) {
    head_norm_rope_kernel<<<dim3(n_heads, rows), kHd / 2, 0, (cudaStream_t) stream>>>(x, n_heads, w, eps, pos0, theta);
    check("q35_df_head_norm_rope");
}

void q35_df_conv(const float* x, const float* dyn, int ldd, const float* base, int side, int rows, int E, int taps,
                 int group, float* out, void* stream) {
    conv_kernel<<<rows, 256, 0, (cudaStream_t) stream>>>(x, dyn, ldd, base, side, E, taps, group, out);
    check("q35_df_conv");
}

void q35_df_ring_store(const float* K, const float* V, int rows, int kvd, int64_t pos0, int ring, uint16_t* Kr,
                       uint16_t* Vr, int64_t* ring_pos, void* stream) {
    if (rows <= 0) return;
    ring_store_kernel<<<rows, 256, 0, (cudaStream_t) stream>>>(K, V, kvd, pos0, ring, (__half*) Kr, (__half*) Vr,
                                                               ring_pos);
    check("q35_df_ring_store");
}

void q35_df_attn(const float* Q, const uint16_t* Kr, const uint16_t* Vr, const int64_t* ring_pos, const float* Kb,
                 const float* Vb, int rows, int n_head, int n_kv, int64_t n_past, int ring, int window, float* out,
                 void* stream) {
    const size_t smem = (size_t) (window + rows) * sizeof(float);
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute(attn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024);
        attr = true;
    }
    attn_kernel<<<dim3(n_head, rows), kHd, smem, (cudaStream_t) stream>>>(
        Q, (const __half*) Kr, (const __half*) Vr, ring_pos, Kb, Vb, rows, n_head, n_kv, n_past, ring, window,
        1.0f / sqrtf((float) kHd), out);
    check("q35_df_attn");
}

int64_t q35_df_topk_scratch(int rows, int k) { return (int64_t) rows * kSlices * k * 8; }

void q35_df_topk(const float* x, int64_t ld, int n, int rows, int k, int* ids, float* vals, void* scratch,
                 void* stream) {
    if (k > kTopKMax) throw std::invalid_argument("q35_df_topk: k");
    const int slice = (n + kSlices - 1) / kSlices;
    int* pi = (int*) scratch;
    float* pv = (float*) (pi + (size_t) rows * kSlices * k);
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute(topk_part_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024);
        attr = true;
    }
    topk_part_kernel<<<dim3(kSlices, rows), 256, (size_t) slice * sizeof(float), (cudaStream_t) stream>>>(
        x, ld, n, k, slice, pi, pv);
    check("q35_df_topk_part");
    topk_merge_kernel<<<rows, 32, (size_t) kSlices * k * 8, (cudaStream_t) stream>>>(pi, pv, k, ids, vals);
    check("q35_df_topk_merge");
}

void q35_df_walk(const int* cand, const float* lg, const float* gate, int k, int rank, const uint16_t* pred_tab,
                 const uint16_t* succ_tab, int anchor, int n, int* tok, float* conf, void* stream) {
    walk_kernel<<<1, rank, 0, (cudaStream_t) stream>>>(cand, lg, gate, k, rank, (const __half*) pred_tab,
                                                       (const __half*) succ_tab, anchor, n, tok, conf);
    check("q35_df_walk");
}

}  // namespace strata::kernels
