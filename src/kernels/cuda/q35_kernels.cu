// src/kernels/cuda/q35_kernels.cu - see include/strata/kernels/q35_kernels.hpp.
#include "strata/kernels/q35_kernels.hpp"
#include "strata/kernels/dense_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// sum over the block (blockDim.x a multiple of 32); every thread gets it
__device__ __forceinline__ float block_sum(float v, float* sh) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) sh[w] = v;
    __syncthreads();
    float s = lane < nw ? sh[lane] : 0.0f;
    return warp_sum(s);
}

// ------------------------------------------------------------------------------------------------ norms
__global__ void rms_rows_kernel(const float* __restrict__ x, int ldx, const float* __restrict__ w,
                                float* __restrict__ of, __half* __restrict__ oh, int cols, float eps) {
    __shared__ float sh[32];
    const float* xr = x + (size_t) blockIdx.x * ldx;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) acc += xr[i] * xr[i];
    const float scale = rsqrtf(block_sum(acc, sh) / (float) cols + eps);
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        const float v = xr[i] * scale * (w ? w[i] : 1.0f);   // (no weight: a plain rms norm)
        if (of) of[(size_t) blockIdx.x * cols + i] = v;
        if (oh) oh[(size_t) blockIdx.x * cols + i] = __float2half(v);
    }
}

__global__ void to_half_kernel(const float* __restrict__ s, __half* __restrict__ d, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = __float2half(s[i]);
}

// ------------------------------------------------------------------------------------------------ GDN
// a thread per channel: the window [s0 s1 s2 | x_t] slides over the T tokens (native_gdn_preprocess.cu's conv_silu)
__global__ void conv_seq_kernel(float* __restrict__ st, const float* __restrict__ raw, int ld,
                                const float* __restrict__ w, float* __restrict__ out, int T, int C, int commit) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float h0 = st[c * 3], h1 = st[c * 3 + 1], h2 = st[c * 3 + 2];
    const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    for (int t = 0; t < T; ++t) {
        const float x = raw[(size_t) t * ld + c];
        float sum = 0.0f;
        sum += h0 * w0;
        sum += h1 * w1;
        sum += h2 * w2;
        sum += x * w3;
        sum = __fadd_rn(sum, 0.0f);
        out[(size_t) t * ld + c] = sum / (1.0f + expf(-sum));
        h0 = h1; h1 = h2; h2 = x;
    }
    if (commit) { st[c * 3] = h0; st[c * 3 + 1] = h1; st[c * 3 + 2] = h2; }
}

// block (row, token), 128 threads: x / sqrt(sum x^2 + eps), as native_gdn_preprocess.cu's l2_norm computes it
__global__ void qk_l2_kernel(float* __restrict__ x, int ld, float eps) {
    __shared__ float sh[32];
    float* r = x + (size_t) blockIdx.y * ld + (size_t) blockIdx.x * 128;
    const float v = r[threadIdx.x];
    const float sum = block_sum(v * v, sh);
    const float scale = rsqrtf(sum / 128.0f + eps / 128.0f);
    r[threadIdx.x] = __fmaf_rn(__fmul_rn(scale, v), 1.0f / sqrtf(128.0f), 0.0f);
}

// conv_seq + qk_l2 + gate_beta in one launch (q35_gdn_prep): block b < C / 128 the conv of channels 128 b .. (one q / k
// head row when b < n_qk: its L2 norm per token in the block, the same arithmetic as qk_l2_kernel), the rest gate_beta's
// elements, 128 a block
__global__ void __launch_bounds__(128) gdn_prep_kernel(float* __restrict__ st, const float* __restrict__ raw, int ld,
                                                       const float* __restrict__ w, float* __restrict__ out, int T, int C,
                                                       int commit, int n_qk, float eps, float* __restrict__ a,
                                                       float* __restrict__ bt, const float* __restrict__ dt,
                                                       const float* __restrict__ ssm_a, int h_v, int ldg) {
    __shared__ float sh[32];
    const int nb = C / 128;
    if ((int) blockIdx.x >= nb) {
        const int k = ((int) blockIdx.x - nb) * 128 + threadIdx.x;
        if (k >= T * h_v) return;
        const int h = k % h_v, i = (k / h_v) * ldg + h;
        const float v = __fadd_rn(a[i], dt[h]);
        const float sp = v > 20.0f ? v : logf(1.0f + expf(v));
        a[i] = sp * ssm_a[h];
        bt[i] = 1.0f / (1.0f + expf(-bt[i]));
        return;
    }
    const int c = blockIdx.x * 128 + threadIdx.x;
    const bool qk = (int) blockIdx.x < n_qk;
    float h0 = st[c * 3], h1 = st[c * 3 + 1], h2 = st[c * 3 + 2];
    const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    for (int t = 0; t < T; ++t) {
        const float x = raw[(size_t) t * ld + c];
        float sum = 0.0f;
        sum += h0 * w0;
        sum += h1 * w1;
        sum += h2 * w2;
        sum += x * w3;
        sum = __fadd_rn(sum, 0.0f);
        float o = sum / (1.0f + expf(-sum));
        if (qk) {
            const float s2 = block_sum(o * o, sh);
            const float scale = rsqrtf(s2 / 128.0f + eps / 128.0f);
            o = __fmaf_rn(__fmul_rn(scale, o), 1.0f / sqrtf(128.0f), 0.0f);
            __syncthreads();   // (sh reused by the next token's sum)
        }
        out[(size_t) t * ld + c] = o;
        h0 = h1; h1 = h2; h2 = x;
    }
    if (commit) { st[c * 3] = h0; st[c * 3 + 1] = h1; st[c * 3 + 2] = h2; }
}

__global__ void gate_beta_kernel(float* __restrict__ a, float* __restrict__ b, const float* __restrict__ dt,
                                 const float* __restrict__ ssm_a, int n, int h_v, int ld) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    const int h = k % h_v, i = (k / h_v) * ld + h;
    const float v = __fadd_rn(a[i], dt[h]);
    const float sp = v > 20.0f ? v : logf(1.0f + expf(v));
    a[i] = sp * ssm_a[h];
    b[i] = 1.0f / (1.0f + expf(-b[i]));
}

// block = value head h, thread = state column j; the column's 128 state values in registers
__global__ void __launch_bounds__(128) gdn_seq_kernel(float* __restrict__ state, const float* __restrict__ x, int ld,
                                                      const float* __restrict__ g, const float* __restrict__ b, int ldg,
                                                      float* __restrict__ o, int T, int h_k, int h_v, int commit) {
    constexpr int S = 128;
    const int h = blockIdx.x, j = threadIdx.x, kh = h % h_k;
    __shared__ float sq[S], sk[S];
    float st[S];
#pragma unroll
    for (int i = 0; i < S; ++i) st[i] = state[((size_t) i * h_v + h) * S + j];
    const float qscale = 1.0f / sqrtf((float) S);
    for (int t = 0; t < T; ++t) {
        const float* xr = x + (size_t) t * ld;
        __syncthreads();
        sq[j] = xr[kh * S + j];
        sk[j] = xr[h_k * S + kh * S + j];
        __syncthreads();
        const float v = xr[2 * h_k * S + h * S + j];
        const float dec = expf(g[(size_t) t * ldg + h]);
        const float beta = b[(size_t) t * ldg + h];
        float skd = 0.0f;
#pragma unroll
        for (int i = 0; i < S; ++i) { st[i] *= dec; skd += st[i] * sk[i]; }
        const float d = (v - skd) * beta;
        float acc = 0.0f;
#pragma unroll
        for (int i = 0; i < S; ++i) { st[i] += sk[i] * d; acc += st[i] * sq[i]; }
        o[(size_t) t * h_v * S + h * S + j] = acc * qscale;
    }
    if (commit) {
#pragma unroll
        for (int i = 0; i < S; ++i) state[((size_t) i * h_v + h) * S + j] = st[i];
    }
}

// block = (value head h, 32 state columns), 4 warps: warp w holds rows [32w, 32w + 32) of its lane's column (32
// registers, no spill; the one-block-a-head form above needs all 128 and spills), the dot products are summed over
// the warps through shared memory in a fixed order.  4x the blocks of gdn_seq_kernel (192 on the 27B), coalesced
// state loads (a warp reads 32 consecutive columns of a row)
__global__ void __launch_bounds__(128) gdn_seq4_kernel(float* __restrict__ state, const float* __restrict__ x, int ld,
                                                       const float* __restrict__ g, const float* __restrict__ b, int ldg,
                                                       float* __restrict__ o, int T, int h_k, int h_v, int commit) {
    constexpr int S = 128, R = 32;
    const int h = blockIdx.x, lane = threadIdx.x & 31, w = threadIdx.x >> 5, kh = h % h_k;
    const int j = blockIdx.y * 32 + lane, i0 = w * R;
    __shared__ float sq[S], sk[S], rs[4][32], ra[4][32];
    float st[R];
#pragma unroll
    for (int i = 0; i < R; ++i) st[i] = state[((size_t) (i0 + i) * h_v + h) * S + j];
    const float qscale = 1.0f / sqrtf((float) S);
    for (int t = 0; t < T; ++t) {
        const float* xr = x + (size_t) t * ld;
        __syncthreads();
        sq[threadIdx.x] = xr[kh * S + threadIdx.x];
        sk[threadIdx.x] = xr[h_k * S + kh * S + threadIdx.x];
        __syncthreads();
        const float v = xr[2 * h_k * S + h * S + j];
        const float dec = expf(g[(size_t) t * ldg + h]);
        const float beta = b[(size_t) t * ldg + h];
        float p = 0.0f;
#pragma unroll
        for (int i = 0; i < R; ++i) { st[i] *= dec; p += st[i] * sk[i0 + i]; }
        rs[w][lane] = p;
        __syncthreads();
        const float skd = ((rs[0][lane] + rs[1][lane]) + rs[2][lane]) + rs[3][lane];
        const float d = (v - skd) * beta;
        float acc = 0.0f;
#pragma unroll
        for (int i = 0; i < R; ++i) { st[i] += sk[i0 + i] * d; acc += st[i] * sq[i0 + i]; }
        ra[w][lane] = acc;
        __syncthreads();
        if (w == 0) o[(size_t) t * h_v * S + h * S + j] = (((ra[0][lane] + ra[1][lane]) + ra[2][lane]) + ra[3][lane]) * qscale;
    }
    if (commit) {
#pragma unroll
        for (int i = 0; i < R; ++i) state[((size_t) (i0 + i) * h_v + h) * S + j] = st[i];
    }
}

// ---- commit of every GDN layer in two launches (a verify window's kept n tokens): the conv states take the last 3
// of [state | raw rows], the recurrence runs gdn_seq4's arithmetic over the stashed conv rows (no output)
__global__ void commit_conv_kernel(const Q35CommitLayer* __restrict__ tab, int n, int C) {
    const Q35CommitLayer& L = tab[blockIdx.y];
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float h[3] = {L.conv_state[c * 3], L.conv_state[c * 3 + 1], L.conv_state[c * 3 + 2]};
    for (int t = 0; t < n; ++t) { h[0] = h[1]; h[1] = h[2]; h[2] = L.raw[(size_t) t * C + c]; }
    L.conv_state[c * 3] = h[0]; L.conv_state[c * 3 + 1] = h[1]; L.conv_state[c * 3 + 2] = h[2];
}
__global__ void __launch_bounds__(128) commit_gdn_kernel(const Q35CommitLayer* __restrict__ tab, int n, int C, int h_k,
                                                         int h_v) {
    constexpr int S = 128, R = 32;
    const Q35CommitLayer& L = tab[blockIdx.z];
    float* __restrict__ state = L.state;
    const float* __restrict__ x = L.conv;
    const float* __restrict__ g = L.ab;
    const float* __restrict__ b = L.ab + h_v;
    const int ldg = 2 * h_v;
    const int h = blockIdx.x, lane = threadIdx.x & 31, w = threadIdx.x >> 5, kh = h % h_k;
    const int j = blockIdx.y * 32 + lane, i0 = w * R;
    __shared__ float sk[S], rs[4][32];
    float st[R];
#pragma unroll
    for (int i = 0; i < R; ++i) st[i] = state[((size_t) (i0 + i) * h_v + h) * S + j];
    for (int t = 0; t < n; ++t) {
        const float* xr = x + (size_t) t * C;
        __syncthreads();
        sk[threadIdx.x] = xr[h_k * S + kh * S + threadIdx.x];
        __syncthreads();
        const float v = xr[2 * h_k * S + h * S + j];
        const float dec = expf(g[(size_t) t * ldg + h]);
        const float beta = b[(size_t) t * ldg + h];
        float p = 0.0f;
#pragma unroll
        for (int i = 0; i < R; ++i) { st[i] *= dec; p += st[i] * sk[i0 + i]; }
        rs[w][lane] = p;
        __syncthreads();
        const float skd = ((rs[0][lane] + rs[1][lane]) + rs[2][lane]) + rs[3][lane];
        const float d = (v - skd) * beta;
#pragma unroll
        for (int i = 0; i < R; ++i) st[i] += sk[i0 + i] * d;
    }
#pragma unroll
    for (int i = 0; i < R; ++i) state[((size_t) (i0 + i) * h_v + h) * S + j] = st[i];
}

// ------------------------------------------------------------------------------------------------ attention prep
// the orthonormal 256-point Walsh-Hadamard transform of s (shared) by 256 threads, in place
__device__ __forceinline__ void fwht256(float* s, int t) {
#pragma unroll
    for (int h = 1; h < 256; h <<= 1) {
        __syncthreads();
        const float a = s[t], b = s[t ^ h];
        __syncthreads();
        s[t] = (t & h) ? b - a : a + b;
    }
    __syncthreads();
    s[t] *= 0.0625f;   // 1 / sqrt(256)
    __syncthreads();
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// block (token, head), D threads: the head's rms * weight, then NeoX rope on the first n_rot dims (native_rope.cu);
// K and V into the cache (FP16, or K8V4: INT8 K, rotated 4-bit V)
__global__ void attn_prep_kernel(const float* __restrict__ q_full, int ldq, const float* __restrict__ k,
                                 const float* __restrict__ v, int ldk, const float* __restrict__ q_norm,
                                 const float* __restrict__ k_norm, float eps, float theta_scale, int n_rot, int p0,
                                 int n_head, int D, float* __restrict__ q_out, Q35Kv kv) {
    __shared__ float sh[32];
    __shared__ float row[256];
    const int t = blockIdx.x, hh = blockIdx.y, d = threadIdx.x, pos = p0 + t;
    const bool is_q = hh < n_head;
    const float* src = is_q ? q_full + (size_t) t * ldq + (size_t) hh * 2 * D : k + (size_t) t * ldk + (size_t) (hh - n_head) * D;
    const float val = src[d];
    const float inv = rsqrtf(block_sum(val * val, sh) / (float) D + eps);
    row[d] = val * inv * (is_q ? q_norm[d] : k_norm[d]);
    __syncthreads();
    float r = row[d];
    const int half_rot = n_rot / 2;
    if (d < n_rot) {
        const int pair = d < half_rot ? d : d - half_rot;
        const float theta = (float) pos * powf(theta_scale, (float) pair);
        const float c = cosf(theta), s = sinf(theta);
        const float a = row[pair], bb = row[pair + half_rot];
        r = d < half_rot ? a * c - bb * s : a * s + bb * c;
    }
    if (is_q) {
        q_out[(size_t) t * n_head * D + (size_t) hh * D + d] = r;
        return;
    }
    const int kvh = hh - n_head;
    const size_t cell = (size_t) kvh * kv.max_ctx + pos;
    float vv = v[(size_t) t * ldk + (size_t) kvh * D + d];
    if (kv.mode == 0) {
        ((__half*) kv.k)[cell * D + d] = __float2half(r);
        ((__half*) kv.v)[cell * D + d] = __float2half(vv);
        return;
    }
    // K8V4: K codes with a scale per 32 (a warp), V rotated then 4-bit codes with a scale per 32
    {
        const float amax = warp_max(fabsf(r));
        const float sc = amax / 127.0f;
        const int q = sc > 0.0f ? max(-127, min(127, __float2int_rn(r / sc))) : 0;
        ((int8_t*) kv.k)[cell * D + d] = (int8_t) q;
        if ((d & 31) == 0) ((__half*) kv.ks)[cell * (D / 32) + d / 32] = __float2half(sc);
    }
    __syncthreads();   // (row is reused)
    row[d] = vv;
    fwht256(row, d);
    vv = row[d];
    {
        const float amax = warp_max(fabsf(vv));
        const float sc = amax / 7.0f;
        const int q = sc > 0.0f ? max(-7, min(7, __float2int_rn(vv / sc))) : 0;
        const int qn = __shfl_xor_sync(0xffffffffu, q, 1);
        if ((d & 1) == 0) ((uint8_t*) kv.v)[cell * (D / 2) + d / 2] = (uint8_t) ((q + 8) | ((qn + 8) << 4));
        if ((d & 31) == 0) ((__half*) kv.vs)[cell * (D / 32) + d / 32] = __float2half(sc);
    }
}

// K8V4's output: per (token, head) rotated back by H, then the output gate (block (token, head), 256 threads)
__global__ void unrotate_gate_kernel(float* __restrict__ out, int n_head, const float* __restrict__ q_full, int ldq) {
    __shared__ float row[256];
    const int t = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    float* o = out + (size_t) t * n_head * 256 + (size_t) h * 256;
    row[d] = o[d];
    fwht256(row, d);
    const float g = q_full[(size_t) t * ldq + (size_t) h * 512 + 256 + d];
    o[d] = row[d] / (1.0f + __expf(-g));
}

// ------------------------------------------------------------------------------------------------ attention
constexpr int AD = 256, AR = 64, AK = 32, AW = 4;
constexpr int LDQ = AD + 8, LDS = AK + 4, LDP = AK + 8;   // (wmma: ldm % 8 halves / 4 floats)
constexpr size_t kAttnSmem = (size_t) AR * LDQ * 2 + 2 * (size_t) AK * LDQ * 2 + (size_t) AW * 16 * LDS * 4 +
                             (size_t) AW * 16 * LDP * 2;

__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + __expf(-x)); }

// 8 halves of a K8V4 K row (8 INT8 codes, the group's scale) / V row (8 4-bit codes): the codes become FP16
// 1024 + c with a byte permute (0x64 high bytes), then (h - offset) * scale in half2 arithmetic
__device__ __forceinline__ uint4 deq_k8(uint2 u, __half s1) {
    const __half2 sc = __half2half2(s1), off = __float2half2_rn(1152.0f);   // 1024 + 128 (the sign flip)
    const uint32_t a = u.x ^ 0x80808080u, b = u.y ^ 0x80808080u;
    uint32_t r[4];
    const uint32_t w[4] = {__byte_perm(a, 0x64u, 0x4140), __byte_perm(a, 0x64u, 0x4342),
                           __byte_perm(b, 0x64u, 0x4140), __byte_perm(b, 0x64u, 0x4342)};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const __half2 h = __hmul2(__hsub2(*(const __half2*) &w[i], off), sc);
        r[i] = *(const uint32_t*) &h;
    }
    return make_uint4(r[0], r[1], r[2], r[3]);
}
__device__ __forceinline__ uint4 deq_v4(uint32_t u, __half s1) {
    const __half2 sc = __half2half2(s1), off = __float2half2_rn(1032.0f);    // 1024 + 8 (the code bias)
    // nibbles in order: byte i holds values 2i (low) and 2i + 1 (high)
    const uint32_t lo = u & 0x0F0F0F0Fu, hi = (u >> 4) & 0x0F0F0F0Fu;
    const uint32_t e = __byte_perm(lo, hi, 0x5140), f = __byte_perm(lo, hi, 0x7362);   // lo0 hi0 lo1 hi1 | lo2..
    uint32_t r[4];
    const uint32_t w[4] = {__byte_perm(e, 0x64u, 0x4140), __byte_perm(e, 0x64u, 0x4342),
                           __byte_perm(f, 0x64u, 0x4140), __byte_perm(f, 0x64u, 0x4342)};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const __half2 h = __hmul2(__hsub2(*(const __half2*) &w[i], off), sc);
        r[i] = *(const uint32_t*) &h;
    }
    return make_uint4(r[0], r[1], r[2], r[3]);
}

// grid (row tiles, n_kv, splits), AW warps of 16 rows; see the header.  gated: the output gate applied here (FP16
// cache); K8V4 leaves it to unrotate_gate_kernel (the output is still rotated)
template <int AR_, int AK_, int AW_>
constexpr size_t attn_smem() {
    return (size_t) AR_ * LDQ * 2 + 2 * (size_t) AK_ * LDQ * 2 + (size_t) AW_ * 16 * (AK_ + 4) * 4 +
           (size_t) AW_ * 16 * (AK_ + 8) * 2;
}
// (the tile shape is a parameter: the 64-row default for prompts, 32-row tiles of 2 warps for verify windows)
template <int MODE, int AR_ = 64, int AK_ = 32, int AW_ = 4>
__global__ void __launch_bounds__(AW_ * 32) attn_tc_kernel(const float* __restrict__ q, Q35Kv kv, int p0, int T, int n_head,
                                                          int G, int max_ctx, int split_len, float scale,
                                                          const float* __restrict__ q_full, int ldq,
                                                          float* __restrict__ out, float* __restrict__ part_o,
                                                          float* __restrict__ part_ml) {
    using namespace nvcuda;
    constexpr int AR = AR_, AK = AK_, AW = AW_, LDS = AK + 4, LDP = AK + 8;
    static_assert(AR == AW * 16, "a warp's 16 rows");
    extern __shared__ __align__(32) unsigned char smem[];
    __half* sQ = (__half*) smem;
    __half* sK = sQ + AR * LDQ;
    __half* sV = sK + AK * LDQ;
    float* sS = (float*) (sV + AK * LDQ);
    __half* sP = (__half*) (sS + AW * 16 * LDS);
    const int kvh = blockIdx.y, split = blockIdx.z, n_split = gridDim.z;
    const int rows = T * G, r0 = blockIdx.x * AR;
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    const int qld = n_head * AD;
    // Q (scaled) -> sQ
    for (int i = tid; i < AR * (AD / 4); i += AW * 32) {
        const int r = i / (AD / 4), d = (i % (AD / 4)) * 4, row = r0 + r;
        float4 f = make_float4(0.f, 0.f, 0.f, 0.f);
        if (row < rows) {
            const int tok = row / G, head = kvh * G + row % G;
            f = *(const float4*) (q + (size_t) tok * qld + (size_t) head * AD + d);
        }
        __half2* dst = (__half2*) (sQ + r * LDQ + d);
        dst[0] = __floats2half2_rn(f.x * scale, f.y * scale);
        dst[1] = __floats2half2_rn(f.z * scale, f.w * scale);
    }
    const int last_row = min(r0 + AR, rows) - 1;
    const int kend = p0 + last_row / G + 1;                        // the block's keys: [0, kend)
    const int ks = split * split_len, ke = min(kend, ks + split_len);
    const int wr0 = r0 + w * 16;
    const bool w_on = wr0 < rows;
    const int wlast = w_on ? p0 + (min(wr0 + 16, rows) - 1) / G : -1;
    const int ri = lane >> 1, hi = lane & 1;
    const int row = wr0 + ri;
    const bool row_ok = row < rows;
    const int pos = row_ok ? p0 + row / G : -1;
    float* Sw = sS + w * 16 * LDS;
    __half* Pw = sP + w * 16 * LDP;
    constexpr float kLazy = 8.0f;
    float m = -INFINITY, l = 0.0f;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fo[AD / 16];
#pragma unroll
    for (int n = 0; n < AD / 16; ++n) wmma::fill_fragment(fo[n], 0.0f);
    // the next tile's K / V (raw: FP16, or K8V4 codes and scales) prefetched into registers while this one computes
    constexpr int PF = AK * (AD / 8) / (AW * 32);
    uint4 pk[PF], pv[PF];
    auto fetch = [&](int k0) {
#pragma unroll
        for (int j = 0; j < PF; ++j) {
            const int i = tid + j * AW * 32, r = i / (AD / 8), d = (i % (AD / 8)) * 8, key = k0 + r;
            pk[j] = pv[j] = make_uint4(0, 0, 0, 0);
            if (key < ke) {
                const size_t cell = (size_t) kvh * max_ctx + key;
                if (MODE == 0) {
                    pk[j] = __ldg((const uint4*) ((const __half*) kv.k + cell * AD + d));
                    pv[j] = __ldg((const uint4*) ((const __half*) kv.v + cell * AD + d));
                } else {   // K: 8 codes + its scale's bits; V: 8 nibbles + its scale's bits
                    const uint2 kc = __ldg((const uint2*) ((const int8_t*) kv.k + cell * AD + d));
                    pk[j] = make_uint4(kc.x, kc.y, __ldg((const unsigned short*) kv.ks + cell * (AD / 32) + d / 32), 0);
                    pv[j] = make_uint4(__ldg((const uint32_t*) ((const uint8_t*) kv.v + cell * (AD / 2) + d / 2)),
                                       __ldg((const unsigned short*) kv.vs + cell * (AD / 32) + d / 32), 0, 0);
                }
            }
        }
    };
    if (ks < ke) fetch(ks);
    for (int k0 = ks; k0 < ke; k0 += AK) {
        __syncthreads();
#pragma unroll
        for (int j = 0; j < PF; ++j) {
            const int i = tid + j * AW * 32, r = i / (AD / 8), d = (i % (AD / 8)) * 8;
            uint4 kv4 = pk[j], vv4 = pv[j];
            if (MODE != 0) {
                kv4 = deq_k8(make_uint2(pk[j].x, pk[j].y), __ushort_as_half((unsigned short) pk[j].z));
                vv4 = deq_v4(pv[j].x, __ushort_as_half((unsigned short) pv[j].y));
            }
            *(uint4*) (sK + r * LDQ + d) = kv4;
            *(uint4*) (sV + r * LDQ + d) = vv4;
        }
        __syncthreads();
        if (k0 + AK < ke) fetch(k0 + AK);
        if (!w_on || k0 > wlast) continue;
        // S (16 x AK) = Q_w . K^T
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[AK / 16];
#pragma unroll
        for (int n = 0; n < AK / 16; ++n) wmma::fill_fragment(acc[n], 0.0f);
#pragma unroll
        for (int kk = 0; kk < AD / 16; ++kk) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fq;
            wmma::load_matrix_sync(fq, sQ + (w * 16) * LDQ + kk * 16, LDQ);
#pragma unroll
            for (int n = 0; n < AK / 16; ++n) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
                wmma::load_matrix_sync(fb, sK + (n * 16) * LDQ + kk * 16, LDQ);
                wmma::mma_sync(acc[n], fq, fb, acc[n]);
            }
        }
#pragma unroll
        for (int n = 0; n < AK / 16; ++n) wmma::store_matrix_sync(Sw + n * 16, acc[n], LDS, wmma::mem_row_major);
        __syncwarp();
        constexpr int KH = AK / 2;
        float sv[KH], mt = -INFINITY;
#pragma unroll
        for (int j = 0; j < KH; ++j) {
            const int c = hi * KH + j, key = k0 + c;
            sv[j] = row_ok && key < ke && key <= pos ? Sw[ri * LDS + c] : -INFINITY;
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
        for (int j = 0; j < KH; ++j) {
            const float pj = sv[j] == -INFINITY ? 0.0f : __expf(sv[j] - m);
            ps += pj;
            Pw[ri * LDP + hi * KH + j] = __float2half(pj);
        }
        l += ps + __shfl_xor_sync(0xffffffffu, ps, 1);
        __syncwarp();
        if (__any_sync(0xffffffffu, need)) {
#pragma unroll
            for (int n = 0; n < AD / 16; ++n) {
                wmma::store_matrix_sync(Sw, fo[n], LDS, wmma::mem_row_major);
                __syncwarp();
#pragma unroll
                for (int j = 0; j < 8; ++j) Sw[ri * LDS + hi * 8 + j] *= corr;
                __syncwarp();
                wmma::load_matrix_sync(fo[n], Sw, LDS, wmma::mem_row_major);
                __syncwarp();
            }
        }
#pragma unroll
        for (int kk = 0; kk < AK / 16; ++kk) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fp;
            wmma::load_matrix_sync(fp, Pw + kk * 16, LDP);
#pragma unroll
            for (int n = 0; n < AD / 16; ++n) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fv;
                wmma::load_matrix_sync(fv, sV + (kk * 16) * LDQ + n * 16, LDQ);
                wmma::mma_sync(fo[n], fp, fv, fo[n]);
            }
        }
        __syncwarp();
    }
    if (!w_on) return;
    const int tok = row_ok ? row / G : 0, head = kvh * G + (row_ok ? row % G : 0);
    const float inv_l = l > 0.0f ? 1.0f / l : 0.0f;
#pragma unroll
    for (int n = 0; n < AD / 16; ++n) {
        wmma::store_matrix_sync(Sw, fo[n], LDS, wmma::mem_row_major);
        __syncwarp();
        if (row_ok) {
            const int d0 = n * 16 + hi * 8;
            if (n_split == 1) {
                float* o = out + (size_t) tok * qld + (size_t) head * AD + d0;
                const float* gt = q_full + (size_t) tok * ldq + (size_t) head * 2 * AD + AD + d0;
#pragma unroll
                for (int j = 0; j < 8; ++j) o[j] = Sw[ri * LDS + hi * 8 + j] * inv_l * (MODE == 0 ? sigm(gt[j]) : 1.0f);
            } else {
                float* o = part_o + (((size_t) split * gridDim.y + kvh) * rows + row) * AD + d0;
#pragma unroll
                for (int j = 0; j < 8; ++j) o[j] = Sw[ri * LDS + hi * 8 + j];
            }
        }
        __syncwarp();
    }
    if (n_split > 1 && row_ok && hi == 0) {
        part_ml[(((size_t) split * gridDim.y + kvh) * rows + row) * 2] = m;
        part_ml[(((size_t) split * gridDim.y + kvh) * rows + row) * 2 + 1] = l;
    }
}

// ---- decode / short verify (the tile kernel above fills 1 of its 4 warps with 6-48 rows and runs 1 block a SM):
// block = (kv head, split, token t), 4 warps, warp w takes the split's keys w, w + 4, ..; lane l the dims 8 l .. 8 l + 7
// of the G rows of token t (CUDA cores: q.k by a warp sum, the online softmax and O in registers), two keys in flight
// a warp.  The warps fold through shared memory; partials in attn_tc_kernel's layout (attn_merge_kernel folds them).
template <int MODE, int G>
__global__ void __launch_bounds__(128) attn_dec_kernel(const float* __restrict__ q, Q35Kv kv, int p0, int n_head,
                                                       int max_ctx, int split_len, float scale,
                                                       const float* __restrict__ q_full, int ldq,
                                                       float* __restrict__ out, float* __restrict__ part_o,
                                                       float* __restrict__ part_ml) {
    __shared__ float sm_m[4][G], sm_l[4][G];
    __shared__ float sm_o[4][G][AD];
    const int kvh = blockIdx.x, split = blockIdx.y, n_split = gridDim.y, t = blockIdx.z, T = gridDim.z;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, n_kv = gridDim.x;
    const int ks = split * split_len, ke = min(p0 + t + 1, ks + split_len);
    const int qld = n_head * AD;
    float qv[G][8];
#pragma unroll
    for (int g = 0; g < G; ++g) {
        const float4* qp = (const float4*) (q + (size_t) t * qld + (size_t) (kvh * G + g) * AD + lane * 8);
        const float4 a = qp[0], b = qp[1];
        qv[g][0] = a.x * scale; qv[g][1] = a.y * scale; qv[g][2] = a.z * scale; qv[g][3] = a.w * scale;
        qv[g][4] = b.x * scale; qv[g][5] = b.y * scale; qv[g][6] = b.z * scale; qv[g][7] = b.w * scale;
    }
    float m[G], l[G], acc[G][8];
#pragma unroll
    for (int g = 0; g < G; ++g) {
        m[g] = -INFINITY; l[g] = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[g][j] = 0.0f;
    }
    auto load = [&](int key, uint4& kr, uint4& vr) {
        const size_t cell = (size_t) kvh * max_ctx + key;
        if (MODE == 0) {
            kr = __ldg((const uint4*) ((const __half*) kv.k + cell * AD + lane * 8));
            vr = __ldg((const uint4*) ((const __half*) kv.v + cell * AD + lane * 8));
        } else {
            const uint2 kc = __ldg((const uint2*) ((const int8_t*) kv.k + cell * AD + lane * 8));
            kr = make_uint4(kc.x, kc.y, __ldg((const unsigned short*) kv.ks + cell * (AD / 32) + lane / 4), 0);
            vr = make_uint4(__ldg((const uint32_t*) ((const uint8_t*) kv.v + cell * (AD / 2) + lane * 4)),
                            __ldg((const unsigned short*) kv.vs + cell * (AD / 32) + lane / 4), 0, 0);
        }
    };
    auto step = [&](uint4 kr, uint4 vr) {
        if (MODE != 0) {
            kr = deq_k8(make_uint2(kr.x, kr.y), __ushort_as_half((unsigned short) kr.z));
            vr = deq_v4(vr.x, __ushort_as_half((unsigned short) vr.y));
        }
        float kf[8], vf[8];
        const __half2* k2 = (const __half2*) &kr;
        const __half2* v2 = (const __half2*) &vr;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 a = __half22float2(k2[j]), b = __half22float2(v2[j]);
            kf[2 * j] = a.x; kf[2 * j + 1] = a.y; vf[2 * j] = b.x; vf[2 * j + 1] = b.y;
        }
#pragma unroll
        for (int g = 0; g < G; ++g) {
            float sv = 0.0f;
#pragma unroll
            for (int j = 0; j < 8; ++j) sv += qv[g][j] * kf[j];
            sv = warp_sum(sv);
            const float mn = fmaxf(m[g], sv), corr = __expf(m[g] - mn), p = __expf(sv - mn);
            l[g] = l[g] * corr + p;
            m[g] = mn;
#pragma unroll
            for (int j = 0; j < 8; ++j) acc[g][j] = acc[g][j] * corr + p * vf[j];
        }
    };
    int key = ks + w;
    for (; key + 4 < ke; key += 8) {
        uint4 k0, v0, k1, v1;
        load(key, k0, v0);
        load(key + 4, k1, v1);
        step(k0, v0);
        step(k1, v1);
    }
    if (key < ke) {
        uint4 k0, v0;
        load(key, k0, v0);
        step(k0, v0);
    }
    if (lane == 0)
#pragma unroll
        for (int g = 0; g < G; ++g) { sm_m[w][g] = m[g]; sm_l[w][g] = l[g]; }
#pragma unroll
    for (int g = 0; g < G; ++g)
#pragma unroll
        for (int j = 0; j < 8; ++j) sm_o[w][g][lane * 8 + j] = acc[g][j];
    __syncthreads();
    const int rows = T * G;
    for (int i = threadIdx.x; i < G * AD; i += 128) {
        const int g = i / AD, d = i % AD;
        float M = -INFINITY;
#pragma unroll
        for (int ww = 0; ww < 4; ++ww) if (sm_l[ww][g] > 0.0f) M = fmaxf(M, sm_m[ww][g]);
        float a = 0.0f, L = 0.0f;
#pragma unroll
        for (int ww = 0; ww < 4; ++ww) {
            if (!(sm_l[ww][g] > 0.0f)) continue;
            const float f = __expf(sm_m[ww][g] - M);
            a += sm_o[ww][g][d] * f;
            L += sm_l[ww][g] * f;
        }
        const int head = kvh * G + g, row = t * G + g;
        if (n_split == 1) {
            const float gt = q_full[(size_t) t * ldq + (size_t) head * 2 * AD + AD + d];
            out[(size_t) t * qld + (size_t) head * AD + d] = (L > 0.0f ? a / L : 0.0f) * (MODE == 0 ? sigm(gt) : 1.0f);
        } else {
            part_o[(((size_t) split * n_kv + kvh) * rows + row) * AD + d] = a;
            if (d == 0) {
                part_ml[(((size_t) split * n_kv + kvh) * rows + row) * 2] = M;
                part_ml[(((size_t) split * n_kv + kvh) * rows + row) * 2 + 1] = L;
            }
        }
    }
}

// block = row, AD threads: fold the splits' partials, then the output gate
__global__ void attn_merge_kernel(const float* __restrict__ part_o, const float* __restrict__ part_ml, int rows,
                                  int n_split, int G, int n_head, const float* __restrict__ q_full, int ldq,
                                  float* __restrict__ out, int p0, int split_len, int gated) {
    const int row = blockIdx.x, d = threadIdx.x;
    const int kvh = blockIdx.y, tok = row / G, head = kvh * G + row % G;
    part_o += (size_t) kvh * rows * AD;
    part_ml += (size_t) kvh * rows * 2;
    const int rows_all = rows * gridDim.y;   // a split's partials: [n_kv][rows]
    const int n_use = min(n_split, (p0 + tok) / split_len + 1);   // the splits that hold keys of this row
    float M = -INFINITY;
    for (int s = 0; s < n_use; ++s) {
        const float l = part_ml[((size_t) s * rows_all + row) * 2 + 1];
        if (l > 0.0f) M = fmaxf(M, part_ml[((size_t) s * rows_all + row) * 2]);
    }
    float a = 0.0f, L = 0.0f;
    for (int s = 0; s < n_use; ++s) {
        const float l = part_ml[((size_t) s * rows_all + row) * 2 + 1];
        if (!(l > 0.0f)) continue;
        const float f = __expf(part_ml[((size_t) s * rows_all + row) * 2] - M);
        a += part_o[((size_t) s * rows_all + row) * AD + d] * f;
        L += l * f;
    }
    const float gt = q_full[(size_t) tok * ldq + (size_t) head * 2 * AD + AD + d];
    out[(size_t) tok * n_head * AD + (size_t) head * AD + d] = (L > 0.0f ? a / L : 0.0f) * (gated ? sigm(gt) : 1.0f);
}

// ------------------------------------------------------------------------------------------------ small matmuls
// block = (output row, 8 tokens), 256 threads along n_in; the 8 sums reduced over the block
__global__ void __launch_bounds__(256) gemv_f32_cols_kernel(const float* __restrict__ W, const float* __restrict__ x,
                                                            int ldx, float* __restrict__ y, int n_in, int n_out, int T) {
    constexpr int NC = 8;
    const int r = blockIdx.x, t0 = blockIdx.y * NC, nc = min(NC, T - t0);
    const float* wr = W + (size_t) r * n_in;
    float acc[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) acc[c] = 0.0f;
    for (int i = threadIdx.x; i < n_in; i += blockDim.x) {
        const float w = wr[i];
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (c < nc) acc[c] += w * x[(size_t) (t0 + c) * ldx + i];
    }
    __shared__ float sh[NC][8];
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        const float v = warp_sum(acc[c]);
        if (lane == 0) sh[c][wp] = v;
    }
    __syncthreads();
    if (threadIdx.x < nc) {
        float v = 0.0f;
        for (int k = 0; k < (int) (blockDim.x >> 5); ++k) v += sh[threadIdx.x][k];
        y[(size_t) (t0 + threadIdx.x) * n_out + r] = v;
    }
}

__global__ void argmax_kernel(const float* __restrict__ x, int64_t ld, int n, int32_t* __restrict__ out) {
    const float* xr = x + (size_t) blockIdx.x * ld;
    float best = -INFINITY;
    int bi = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = xr[i];
        if (v > best || (v == best && i < bi)) { best = v; bi = i; }
    }
    __shared__ float sv[32];
    __shared__ int si[32];
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, best, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
    }
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    if (lane == 0) { sv[w] = best; si[w] = bi; }
    __syncthreads();
    if (w == 0) {
        best = lane < (int) (blockDim.x >> 5) ? sv[lane] : -INFINITY;
        bi = lane < (int) (blockDim.x >> 5) ? si[lane] : 0x7fffffff;
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, best, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
        }
        if (lane == 0) out[blockIdx.x] = bi;
    }
}

__global__ void add_kernel(float* __restrict__ y, const float* __restrict__ x, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += x[i];
}

__global__ void swiglu_rows_kernel(const float* __restrict__ g, const float* __restrict__ u, int ldg,
                                   float* __restrict__ o, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float a = g[(size_t) blockIdx.y * ldg + i];
    o[(size_t) blockIdx.y * n + i] = (a / (1.0f + __expf(-a))) * u[(size_t) blockIdx.y * ldg + i];
}

unsigned blocks(int64_t n, int b) { return (unsigned) ((n + b - 1) / b); }

}  // namespace

void q35_rms_rows(const float* x, int ldx, const float* w, float* out_f32, uint16_t* out_f16, int rows, int cols,
                  float eps, void* stream) {
    rms_rows_kernel<<<rows, 256, 0, (cudaStream_t) stream>>>(x, ldx, w, out_f32, (__half*) out_f16, cols, eps);
    check("q35_rms_rows");
}

void q35_to_half(const float* src, uint16_t* dst, int64_t n, void* stream) {
    to_half_kernel<<<blocks(n, 256), 256, 0, (cudaStream_t) stream>>>(src, (__half*) dst, n);
    check("q35_to_half");
}

void q35_conv_seq(float* conv_state, const float* raw, int ld, const float* w, float* out, int T, int C, bool commit,
                  void* stream) {
    conv_seq_kernel<<<blocks(C, 128), 128, 0, (cudaStream_t) stream>>>(conv_state, raw, ld, w, out, T, C, commit);
    check("q35_conv_seq");
}

void q35_qk_l2(float* x, int ld, int T, int n_rows, float eps, void* stream) {
    qk_l2_kernel<<<dim3(n_rows, T), 128, 0, (cudaStream_t) stream>>>(x, ld, eps);
    check("q35_qk_l2");
}

bool q35_gdn_prep(float* conv_state, const float* raw, int ld, const float* w, float* out, int T, int C, bool commit,
                  int n_qk_rows, float eps, float* alpha, float* beta, int ldg, const float* dt, const float* ssm_a, int h_v,
                  void* stream) {
    if (C % 128 != 0 || n_qk_rows * 128 > C) return false;
    const int nb = C / 128 + (T * h_v + 127) / 128;
    gdn_prep_kernel<<<nb, 128, 0, (cudaStream_t) stream>>>(conv_state, raw, ld, w, out, T, C, commit, n_qk_rows, eps, alpha,
                                                           beta, dt, ssm_a, h_v, ldg);
    check("q35_gdn_prep");
    return true;
}

void q35_gate_beta(float* alpha, float* beta, int ld, const float* dt, const float* ssm_a, int T, int h_v, void* stream) {
    gate_beta_kernel<<<blocks((int64_t) T * h_v, 256), 256, 0, (cudaStream_t) stream>>>(alpha, beta, dt, ssm_a,
                                                                                       T * h_v, h_v, ld);
    check("q35_gate_beta");
}

void q35_gdn_seq(float* state, const float* x, int ld, const float* g, const float* b, int ldg, float* o, int T,
                 int h_k, int h_v, bool commit, void* stream) {
    if (T <= 0) return;
    static const bool old = std::getenv("STRATA_DENSE_GDN_OLD") && std::atoi(std::getenv("STRATA_DENSE_GDN_OLD")) != 0;
    if (old)
        gdn_seq_kernel<<<h_v, 128, 0, (cudaStream_t) stream>>>(state, x, ld, g, b, ldg, o, T, h_k, h_v, commit);
    else
        gdn_seq4_kernel<<<dim3(h_v, 4), 128, 0, (cudaStream_t) stream>>>(state, x, ld, g, b, ldg, o, T, h_k, h_v, commit);
    check("q35_gdn_seq");
}

void q35_commit_all(const Q35CommitLayer* tab_dev, int n_layers, int n, int C, int h_k, int h_v, void* stream) {
    if (n <= 0 || n_layers <= 0) return;
    commit_conv_kernel<<<dim3(blocks(C, 128), n_layers), 128, 0, (cudaStream_t) stream>>>(tab_dev, n, C);
    check("q35_commit_conv");
    commit_gdn_kernel<<<dim3(h_v, 4, n_layers), 128, 0, (cudaStream_t) stream>>>(tab_dev, n, C, h_k, h_v);
    check("q35_commit_gdn");
}

void q35_gdn_out(const float* o, const float* z, const float* w, float* y, int rows, float eps, void* stream) {
    dense_gdn_out_norm_silu(o, z, w, y, rows, 128, eps, stream);
}

uint64_t q35_kv_bytes(int mode, int n_kv, int D, int max_ctx) {
    const uint64_t cells = (uint64_t) n_kv * max_ctx;
    return mode == 0 ? cells * D * 2 * 2 : cells * (D + D / 32 * 2 + D / 2 + D / 32 * 2);
}

void q35_attn_prep(const float* q_full, int ldq, const float* k, const float* v, int ldk, const float* q_norm,
                   const float* k_norm, float eps, float rope_base, int n_rot, int p0, int T, int n_head, int n_kv,
                   int D, float* q_out, const Q35Kv& kv, void* stream) {
    if (D != 256 || n_rot > D) throw std::invalid_argument("q35_attn_prep: head size 256");
    const float theta_scale = powf(rope_base, -2.0f / n_rot);
    attn_prep_kernel<<<dim3(T, n_head + n_kv), D, 0, (cudaStream_t) stream>>>(
        q_full, ldq, k, v, ldk, q_norm, k_norm, eps, theta_scale, n_rot, p0, n_head, D, q_out, kv);
    check("q35_attn_prep");
}

namespace {
constexpr int kSplitLen = 128;   // the shortest split (a decode step at a short context still spreads)
int g_dec_max = -1;
int g_attn_var = -1;
}

// the CUDA-core decode kernel serves windows of up to this many tokens (STRATA_DENSE_ATTN_DEC, default below)
int q35_attn_dec_max_tokens() {
    if (g_dec_max < 0) {
        const char* e = std::getenv("STRATA_DENSE_ATTN_DEC");
        g_dec_max = e ? std::atoi(e) : 1;   // (one token: 28K plain decode 29.8 -> 30.8 tok/s; 2+ rows: the tile kernel)
    }
    return g_dec_max;
}
void q35_set_attn_dec_max_tokens(int n) { g_dec_max = n; }
void q35_set_attn_var(int v) { g_attn_var = v; }

int64_t q35_attn_scratch_floats(int T, int n_head, int D, int max_ctx) {
    const int64_t splits = (max_ctx + kSplitLen - 1) / kSplitLen;
    return splits * (int64_t) T * n_head * (D + 2);
}

void q35_attn(const float* q, const Q35Kv& kv, int p0, int T, int n_head, int n_kv, int D, const float* q_full,
              int ldq, float* out, float* scratch, int splits, void* stream) {
    if (D != AD || n_head % n_kv != 0 || n_head / n_kv > 8) throw std::invalid_argument("q35_attn: geometry");
    static const bool attr_ok =
        cudaFuncSetAttribute(attn_tc_kernel<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) kAttnSmem) == cudaSuccess &&
        cudaFuncSetAttribute(attn_tc_kernel<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) kAttnSmem) == cudaSuccess;
    if (!attr_ok) throw std::runtime_error("q35_attn: shared memory attribute");
    const int max_ctx = kv.max_ctx;
    const int G = n_head / n_kv, rows = T * G;
    // verify windows (2-8 tokens): STRATA_DENSE_ATTN_VAR 1 = 32-row tiles of 2 warps (32 keys a step), 2 = the same with
    // 16 keys a step (2 blocks a SM); 0 = the 64-row tile
    static int var = -1;
    // (attn_bench 64K FP16 a layer, variant 2: 2 rows 0.674 -> 0.450 ms, 4 rows 0.698 -> 0.542, 8 rows 0.753 -> 0.876;
    // the engine, 27B MTP3 at 28K, two cooled rounds: verify 40.1 / 40.3 -> 39.3 / 39.3 ms, 58.4 / 58.2 -> 59.5 / 59.5
    // tok/s - the default for 2-4 tokens (-2); 0: the 64-row tile)
    if (var < 0) var = std::getenv("STRATA_DENSE_ATTN_VAR") ? std::atoi(std::getenv("STRATA_DENSE_ATTN_VAR")) : -2;
    const int vsel = g_attn_var >= 0 ? g_attn_var : var;
    const int v = vsel == -2 ? (T >= 2 && T <= 4 ? 2 : 0) : (T >= 2 && T <= 8) ? vsel : 0;
    const int AR_v = v ? 32 : AR, AK_v = v == 2 ? 16 : AK, AW_v = v ? 2 : AW;
    const int tiles = (rows + AR_v - 1) / AR_v;
    const int n_keys = p0 + T;
    if (G == 6 && splits <= 0 && T <= q35_attn_dec_max_tokens()) {
        // the CUDA-core decode kernel: ~4 blocks a SM over (kv head, token), at least 64 keys a split
        const int cap = (int) (q35_attn_scratch_floats(T, n_head, D, max_ctx) / ((int64_t) rows * n_kv * (D + 2)));
        int n_split = std::max(1, std::min({320 / (n_kv * T), (n_keys + 63) / 64, cap}));
        const int split_len = (n_keys + n_split - 1) / n_split;
        n_split = (n_keys + split_len - 1) / split_len;
        if ((int64_t) n_split * rows * n_kv * (D + 2) > q35_attn_scratch_floats(T, n_head, D, max_ctx))
            throw std::runtime_error("q35_attn: scratch too small");
        float* part_o = scratch;
        float* part_ml = scratch + (size_t) n_split * rows * n_kv * D;
        const dim3 grid(n_kv, n_split, T);
        const float sc = 1.0f / sqrtf((float) D);
        if (kv.mode == 0)
            attn_dec_kernel<0, 6><<<grid, 128, 0, (cudaStream_t) stream>>>(q, kv, p0, n_head, max_ctx, split_len, sc, q_full,
                                                                          ldq, out, part_o, part_ml);
        else
            attn_dec_kernel<1, 6><<<grid, 128, 0, (cudaStream_t) stream>>>(q, kv, p0, n_head, max_ctx, split_len, sc, q_full,
                                                                          ldq, out, part_o, part_ml);
        check("q35_attn_dec");
        if (n_split > 1) {
            attn_merge_kernel<<<dim3(rows, n_kv), D, 0, (cudaStream_t) stream>>>(part_o, part_ml, rows, n_split, G, n_head,
                                                                                 q_full, ldq, out, p0, split_len,
                                                                                 kv.mode == 0);
            check("q35_attn_merge");
        }
        if (kv.mode != 0) {
            unrotate_gate_kernel<<<dim3(T, n_head), 256, 0, (cudaStream_t) stream>>>(out, n_head, q_full, ldq);
            check("q35_attn_unrotate");
        }
        return;
    }
    // splits: enough blocks to fill the GPU when the rows are few
    int n_split = splits;
    if (n_split <= 0) {
        const int want = (v == 2 ? 320 : 160) / std::max(1, tiles * n_kv);   // (variant 2: two blocks a SM)
        n_split = std::max(1, std::min(want, (n_keys + kSplitLen - 1) / kSplitLen));
    }
    const int split_len = n_split == 1 ? n_keys : ((n_keys + n_split - 1) / n_split + AK - 1) / AK * AK;   // (AK: 32)
    n_split = (n_keys + split_len - 1) / split_len;
    if ((int64_t) n_split * rows * n_kv * (D + 2) > q35_attn_scratch_floats(T, n_head, D, max_ctx))
        throw std::runtime_error("q35_attn: scratch too small");
    float* part_o = scratch;
    float* part_ml = scratch + (size_t) n_split * rows * n_kv * D;
    const dim3 grid(tiles, n_kv, n_split);
    const float sc = 1.0f / sqrtf((float) D);
    auto go = [&](auto kern, size_t smem, int threads) {
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem);
        kern<<<grid, threads, smem, (cudaStream_t) stream>>>(q, kv, p0, T, n_head, G, max_ctx, split_len, sc, q_full, ldq,
                                                             out, part_o, part_ml);
    };
    if (v == 1) {
        if (kv.mode == 0) go(attn_tc_kernel<0, 32, 32, 2>, attn_smem<32, 32, 2>(), 64);
        else go(attn_tc_kernel<1, 32, 32, 2>, attn_smem<32, 32, 2>(), 64);
    } else if (v == 2) {
        if (kv.mode == 0) go(attn_tc_kernel<0, 32, 16, 2>, attn_smem<32, 16, 2>(), 64);
        else go(attn_tc_kernel<1, 32, 16, 2>, attn_smem<32, 16, 2>(), 64);
    } else if (kv.mode == 0) {
        attn_tc_kernel<0><<<grid, AW * 32, kAttnSmem, (cudaStream_t) stream>>>(
            q, kv, p0, T, n_head, G, max_ctx, split_len, sc, q_full, ldq, out, part_o, part_ml);
    } else {
        attn_tc_kernel<1><<<grid, AW * 32, kAttnSmem, (cudaStream_t) stream>>>(
            q, kv, p0, T, n_head, G, max_ctx, split_len, sc, q_full, ldq, out, part_o, part_ml);
    }
    check("q35_attn");
    if (n_split > 1) {
        attn_merge_kernel<<<dim3(rows, n_kv), D, 0, (cudaStream_t) stream>>>(part_o, part_ml, rows, n_split, G, n_head,
                                                                             q_full, ldq, out, p0, split_len,
                                                                             kv.mode == 0);
        check("q35_attn_merge");
    }
    if (kv.mode != 0) {
        unrotate_gate_kernel<<<dim3(T, n_head), 256, 0, (cudaStream_t) stream>>>(out, n_head, q_full, ldq);
        check("q35_attn_unrotate");
    }
}

void q35_gemv_f32_cols(const float* W, const float* x, int ldx, float* y, int n_in, int n_out, int T, void* stream) {
    gemv_f32_cols_kernel<<<dim3(n_out, (T + 7) / 8), 256, 0, (cudaStream_t) stream>>>(W, x, ldx, y, n_in, n_out, T);
    check("q35_gemv_f32_cols");
}

void q35_argmax_rows(const float* x, int64_t ld, int n, int T, int32_t* out, void* stream) {
    argmax_kernel<<<T, 1024, 0, (cudaStream_t) stream>>>(x, ld, n, out);
    check("q35_argmax_rows");
}

void q35_add(float* y, const float* x, int64_t n, void* stream) {
    add_kernel<<<blocks(n, 256), 256, 0, (cudaStream_t) stream>>>(y, x, n);
    check("q35_add");
}

void q35_swiglu_rows(const float* g, const float* u, int ldg, float* out, int n, int T, void* stream) {
    swiglu_rows_kernel<<<dim3(blocks(n, 256), T), 256, 0, (cudaStream_t) stream>>>(g, u, ldg, out, n);
    check("q35_swiglu_rows");
}

}  // namespace strata::kernels
