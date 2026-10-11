// src/kernels/cuda/glm_gqa.cu - hy_v3's grouped-query attention for one decode token (glm_fast.hpp: gqa_prep,
// gqa_attn).  The arithmetic is llama.cpp's hy-v3 graph (src/models/hy-v3.cpp): per-head RMS norms on q and k
// (llm_build_norm over head_dim), NeoX rope over the whole head (ggml-cuda/rope.cu's rope_neox: theta = pos *
// base^(-2i/n_dims), the pair (i, i + n_dims/2)), K and V kept in an FP16 cache, softmax(q.k * 1/sqrt(head_dim)) . v.
#include "strata/kernels/glm_fast.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace strata::kernels::glmf {
namespace {

constexpr int kD = 128;           // head_dim (hy3_geometry requires it)
constexpr int kChunk = 256;       // positions per split of the attention
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

// one block of kD threads per head: blocks [0, n_head) the q heads, [n_head, n_head + n_kv) the k heads, then the
// v heads.  q: norm + rope in place (f32); k: norm + rope into the K cache row at pos; v: into the V cache row.
__global__ void __launch_bounds__(kD) gqa_prep_kernel(float* __restrict__ q, const float* __restrict__ k,
                                                      const float* __restrict__ v, const float* __restrict__ qn,
                                                      const float* __restrict__ kn, float eps, int pos, float theta_scale,
                                                      int n_head, int n_kv, __half* __restrict__ kc,
                                                      __half* __restrict__ vc) {
    __shared__ float sy[kD];
    __shared__ float sred[kD / 32];
    const int b = blockIdx.x, t = threadIdx.x;
    if (b >= n_head + n_kv) {                                  // a v head: copy
        const int h = b - n_head - n_kv;
        vc[((size_t) pos * n_kv + h) * kD + t] = __float2half(v[(size_t) h * kD + t]);
        return;
    }
    const bool is_q = b < n_head;
    const int h = is_q ? b : b - n_head;
    const float x = is_q ? q[(size_t) h * kD + t] : k[(size_t) h * kD + t];
    float ss = wsum(x * x);
    if ((t & 31) == 0) sred[t >> 5] = ss;
    __syncthreads();
    ss = 0.0f;
#pragma unroll
    for (int i = 0; i < kD / 32; ++i) ss += sred[i];
    const float y = x * rsqrtf(ss / (float) kD + eps) * (is_q ? qn[t] : kn[t]);
    sy[t] = y;
    __syncthreads();
    if (t >= kD / 2) return;
    // rope_neox: x0 = y[i], x1 = y[i + n_dims/2], theta = pos * theta_scale^i
    const float theta = (float) pos * powf(theta_scale, (float) t);
    const float c = cosf(theta), s = sinf(theta);
    const float x0 = sy[t], x1 = sy[t + kD / 2];
    const float o0 = x0 * c - x1 * s, o1 = x0 * s + x1 * c;
    if (is_q) {
        q[(size_t) h * kD + t] = o0;
        q[(size_t) h * kD + t + kD / 2] = o1;
    } else {
        __half* row = kc + ((size_t) pos * n_kv + h) * kD;
        row[t] = __float2half(o0);
        row[t + kD / 2] = __float2half(o1);
    }
}

// split-K decode attention: block (kv head, split) takes positions [split * kChunk, +kChunk) for the G query heads
// of that K/V head; writes per (head, split) the running max m, the sum l and the unnormalised output o[kD].
// 256 threads = 8 warps.  Scores: a warp per position, each lane 4 of the head's 128 dims (one 8-byte K load).
__global__ void __launch_bounds__(256) gqa_attn_kernel(const float* __restrict__ q, const __half* __restrict__ kc,
                                                       const __half* __restrict__ vc, int n_pos, int n_head, int n_kv,
                                                       float scale, float* __restrict__ part) {
    __shared__ float sq[kMaxGroup * kD];
    __shared__ float sp[kMaxGroup * kChunk];
    __shared__ float sm[kMaxGroup], sl[kMaxGroup];
    const int kvh = blockIdx.x, split = blockIdx.y, G = n_head / n_kv;
    const int t = threadIdx.x, lane = t & 31, w = t >> 5;
    const int p0 = split * kChunk, np = min(kChunk, n_pos - p0);
    for (int i = t; i < G * kD; i += blockDim.x) sq[i] = q[(size_t) (kvh * G) * kD + i] * scale;
    __syncthreads();
    // 1. scores
    for (int j = w; j < np; j += 8) {
        const __half2* kr = (const __half2*) (kc + ((size_t) (p0 + j) * n_kv + kvh) * kD) + 2 * lane;
        const float2 k01 = __half22float2(kr[0]), k23 = __half22float2(kr[1]);
        for (int g = 0; g < G; ++g) {
            const float* qg = sq + g * kD + 4 * lane;
            float d = qg[0] * k01.x + qg[1] * k01.y + qg[2] * k23.x + qg[3] * k23.y;
            d = wsum(d);
            if (lane == 0) sp[g * kChunk + j] = d;
        }
    }
    __syncthreads();
    // 2. per head: max, exp, sum (warp g handles head g, and g + 8)
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
    // 3. o[g][d] = sum_j p[g][j] v[j][d]: thread t owns dims 4 (t % 32) .. +3 of heads g = t / 32, t / 32 + 8
    const int d4 = 4 * lane;
    for (int g = w; g < G; g += 8) {
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        for (int j = 0; j < np; ++j) {
            const __half2* vr = (const __half2*) (vc + ((size_t) (p0 + j) * n_kv + kvh) * kD + d4);
            const float2 v01 = __half22float2(vr[0]), v23 = __half22float2(vr[1]);
            const float pj = sp[g * kChunk + j];
            a0 += pj * v01.x;
            a1 += pj * v01.y;
            a2 += pj * v23.x;
            a3 += pj * v23.y;
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

// combine the splits of each head: out[h][d] = sum_s e^(m_s - M) o_s[d] / sum_s e^(m_s - M) l_s
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

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) std::fprintf(stderr, "glm_gqa %s: %s\n", what, cudaGetErrorString(e));
}

}  // namespace

size_t gqa_attn_scratch_floats(int n_head, int max_pos) {
    const int n_split = (max_pos + kChunk - 1) / kChunk;
    return (size_t) n_head * n_split * (kD + 2) + (size_t) n_head * kD;
}

void gqa_prep(float* q, const float* k, const float* v, const float* q_norm, const float* k_norm, float eps, int pos,
              float rope_base, int n_head, int n_kv, void* k_cache, void* v_cache, cudaStream_t s) {
    const float theta_scale = powf(rope_base, -2.0f / (float) kD);
    gqa_prep_kernel<<<n_head + 2 * n_kv, kD, 0, s>>>(q, k, v, q_norm, k_norm, eps, pos, theta_scale, n_head, n_kv,
                                                    (__half*) k_cache, (__half*) v_cache);
    check("gqa_prep");
}

bool gqa_attn(const float* q, const void* k_cache, const void* v_cache, int n_pos, int n_head, int n_kv,
              float* scratch, void* out_q8_1, cudaStream_t s) {
    if (n_head % n_kv != 0 || n_head / n_kv > kMaxGroup || n_pos < 1) return false;
    const int n_split = (n_pos + kChunk - 1) / kChunk;
    float* part = scratch;
    float* out = scratch + (size_t) n_head * n_split * (kD + 2);
    gqa_attn_kernel<<<dim3(n_kv, n_split), 256, 0, s>>>(q, (const __half*) k_cache, (const __half*) v_cache, n_pos,
                                                       n_head, n_kv, 1.0f / sqrtf((float) kD), part);
    check("gqa_attn");
    gqa_combine_kernel<<<n_head, kD, 0, s>>>(part, n_split, out);
    check("gqa_combine");
    quantize_q8_1(out, out_q8_1, n_head * kD, s);
    return true;
}

}  // namespace strata::kernels::glmf
