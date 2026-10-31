// src/nh/nh_kernels.cu - maya-nh's kernels (nh_kernels.cuh)
#include "nh_kernels.cuh"

#include <cmath>
#include <cstdio>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace nh::k {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) std::fprintf(stderr, "nh %s: %s\n", what, cudaGetErrorString(e));
}

__device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }
__device__ __forceinline__ float softplus(float x) { return x > 20.0f ? x : log1pf(expf(x)); }

// ------------------------------------------------------------------ the causal depthwise conv
// one thread a channel: its four taps, the tail's three inputs, then each token in order
__global__ void conv_silu_kernel(const float* x, int ldx, int T, int C, const float* w, const float* bias, float* tail,
                                 float* out) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const float w0 = w[(size_t) c * 4], w1 = w[(size_t) c * 4 + 1], w2 = w[(size_t) c * 4 + 2],
                w3 = w[(size_t) c * 4 + 3], b = bias[c];
    float a0 = tail[(size_t) c * 3], a1 = tail[(size_t) c * 3 + 1], a2 = tail[(size_t) c * 3 + 2];
    for (int t = 0; t < T; ++t) {
        const float xt = x[(size_t) t * ldx + c];
        // (the sum in the order transformers' / tools/nh/ref.py's: bias, then the taps from the oldest)
        float v = b;
        v += w0 * a0;
        v += w1 * a1;
        v += w2 * a2;
        v += w3 * xt;
        out[(size_t) t * C + c] = silu(v);
        a0 = a1;
        a1 = a2;
        a2 = xt;
    }
    tail[(size_t) c * 3] = a0;
    tail[(size_t) c * 3 + 1] = a1;
    tail[(size_t) c * 3 + 2] = a2;
}

// ------------------------------------------------------------------ the selective scan
// one block a head, 256 threads: thread (p = tid / 4, q = tid % 4) holds S[p][q 32 .. q 32 + 31] in registers; each
// token: S = dA S + (dt x_p) B, y_p = sum_n S C (the four threads' parts added), + D x_p.  B and C through shared memory.
constexpr int kScanP = 64, kScanN = 128, kScanQ = 4, kScanPer = kScanN / kScanQ;
__global__ void __launch_bounds__(256) ssm_scan_kernel(const float* xbc, int T, int H, int G, const float* dt,
                                                       int lddt, const float* dt_bias, const float* A, const float* D,
                                                       float* state, float* y) {
    const int h = blockIdx.x, tid = threadIdx.x, p = tid >> 2, q = tid & 3;
    const int HP = H * kScanP, ld = HP + 2 * G * kScanN, g = h / (H / G);
    __shared__ float bs[kScanN], cs[kScanN];
    float S[kScanPer];
    float* st = state + ((size_t) h * kScanP + p) * kScanN + q * kScanPer;
#pragma unroll
    for (int i = 0; i < kScanPer; ++i) S[i] = st[i];
    const float a = A[h], dtb = dt_bias[h], dd = D[h];
    for (int t = 0; t < T; ++t) {
        const float* row = xbc + (size_t) t * ld;
        __syncthreads();
        if (tid < kScanN) bs[tid] = row[HP + g * kScanN + tid];
        else cs[tid - kScanN] = row[HP + G * kScanN + g * kScanN + tid - kScanN];
        __syncthreads();
        const float d = softplus(dt[(size_t) t * lddt + h] + dtb);
        const float dA = expf(d * a);
        const float xp = row[h * kScanP + p];
        const float dx = d * xp;
        float acc = 0.0f;
#pragma unroll
        for (int i = 0; i < kScanPer; ++i) {
            S[i] = S[i] * dA + dx * bs[q * kScanPer + i];
            acc = fmaf(S[i], cs[q * kScanPer + i], acc);
        }
        acc += __shfl_xor_sync(0xffffffffu, acc, 1);
        acc += __shfl_xor_sync(0xffffffffu, acc, 2);
        if (q == 0) y[(size_t) t * HP + h * kScanP + p] = acc + dd * xp;
    }
#pragma unroll
    for (int i = 0; i < kScanPer; ++i) st[i] = S[i];
}

// ------------------------------------------------------------------ the gated RMSNorm
// one block of 256 threads a (token, group)
__global__ void __launch_bounds__(256) gated_norm_kernel(const float* y, const float* z, int ldz, int n, int group,
                                                         const float* w, float eps, float* out, uint16_t* out16) {
    const int t = blockIdx.x, gi = blockIdx.y, tid = threadIdx.x;
    const size_t o = (size_t) gi * group;
    float ss = 0.0f;
    for (int i = tid; i < group; i += 256) {
        const float v = y[(size_t) t * n + o + i] * silu(z[(size_t) t * ldz + o + i]);
        ss = fmaf(v, v, ss);
    }
    __shared__ float red[8];
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, m);
    if ((tid & 31) == 0) red[tid >> 5] = ss;
    __syncthreads();
    if (tid < 32) {
        float v = tid < 8 ? red[tid] : 0.0f;
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
        if (tid == 0) red[0] = v;
    }
    __syncthreads();
    const float r = rsqrtf(red[0] / (float) group + eps);
    for (int i = tid; i < group; i += 256) {
        const float v = y[(size_t) t * n + o + i] * silu(z[(size_t) t * ldz + o + i]) * r * w[o + i];
        if (out) out[(size_t) t * n + o + i] = v;
        if (out16) out16[(size_t) t * n + o + i] = __half_as_ushort(__float2half_rn(v));
    }
}

// ------------------------------------------------------------------ the router
// one warp a token, each lane 32 experts at most (E <= 1024): K rounds of the warp's argmax of s + bias
__device__ __forceinline__ bool better(float a, int ia, float b, int ib) { return a > b || (a == b && ia < ib); }
__global__ void route_kernel(const float* logits, const float* bias, int T, int E, int K, float scale, int* top,
                             float* w) {
    const int t = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (t >= T) return;
    constexpr int kPer = 32;
    float sg[kPer], ch[kPer];
    const int per = (E + 31) / 32;
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
        const int e = lane + 32 * i;
        const bool ok = i < per && e < E;
        sg[i] = ok ? 1.0f / (1.0f + expf(-logits[(size_t) t * E + e])) : 0.0f;
        ch[i] = ok ? sg[i] + bias[e] : -INFINITY;
    }
    float sum = 0.0f;
    float picked_w[32];
    int picked[32];
    for (int r = 0; r < K; ++r) {
        float bv = -INFINITY, bsg = 0.0f;
        int bi = 0x7fffffff, bslot = -1;
#pragma unroll
        for (int i = 0; i < kPer; ++i) {
            const int e = lane + 32 * i;
            if (i < per && e < E && better(ch[i], e, bv, bi)) {
                bv = ch[i];
                bi = e;
                bsg = sg[i];
                bslot = i;
            }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            const float os = __shfl_xor_sync(0xffffffffu, bsg, o);
            if (better(ov, oi, bv, bi)) {
                bv = ov;
                bi = oi;
                bsg = os;
            }
        }
        // the winner's lane takes it out of the running
        if (bi != 0x7fffffff && (bi & 31) == lane) {
#pragma unroll
            for (int i = 0; i < kPer; ++i)
                if (lane + 32 * i == bi) ch[i] = -INFINITY;
        }
        (void) bslot;
        picked[r] = bi;
        picked_w[r] = bsg;
        sum += bsg;
    }
    if (lane == 0) {
        const float inv = scale / (sum + 1e-20f);
        for (int r = 0; r < K; ++r) {
            top[(size_t) t * K + r] = picked[r];
            w[(size_t) t * K + r] = picked_w[r] * inv;
        }
    }
}

// ------------------------------------------------------------------ attention's rows
__global__ void kv_store_kernel(const float* k, const float* v, const int* pos, int Hkv, uint16_t* kc, uint16_t* vc) {
    const int t = blockIdx.x, i = threadIdx.x;   // (Hkv x 128 threads)
    const size_t src = (size_t) t * Hkv * 128 + i, dst = (size_t) pos[t] * Hkv * 128 + i;
    kc[dst] = __bfloat16_as_ushort(__float2bfloat16_rn(k[src]));
    vc[dst] = __bfloat16_as_ushort(__float2bfloat16_rn(v[src]));
}

__global__ void kept_all_kernel(const int* pos, int T, int Hkv, int nblk, int blk, int* kept) {
    const int t = blockIdx.x, g = blockIdx.y;
    const int last = pos[t] / blk;
    int* out = kept + ((size_t) g * T + t) * nblk;
    for (int b = threadIdx.x; b < nblk; b += blockDim.x) out[b] = b <= last ? b : -1;
}

__global__ void relu2_kernel(const float* u, size_t n, float* hf, uint16_t* h16) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float r = fmaxf(u[i], 0.0f);
    const float v = r * r;
    if (hf) hf[i] = v;
    if (h16) h16[i] = __half_as_ushort(__float2half_rn(v));
}

__global__ void __launch_bounds__(256) layer_norm_kernel(const float* x, const float* w, int D, float eps, float* out) {
    const float* r = x + (size_t) blockIdx.x * D;
    __shared__ float red[8];
    __shared__ float stat;
    float a = 0.0f;
    for (int i = threadIdx.x; i < D; i += 256) a += r[i];
    for (int m = 16; m > 0; m >>= 1) a += __shfl_xor_sync(0xffffffffu, a, m);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = a;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.0f;
        for (int i = 0; i < 8; ++i) t += red[i];
        stat = t / (float) D;
    }
    __syncthreads();
    const float mean = stat;
    float v = 0.0f;
    for (int i = threadIdx.x; i < D; i += 256) v = fmaf(r[i] - mean, r[i] - mean, v);
    for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.0f;
        for (int i = 0; i < 8; ++i) t += red[i];
        stat = rsqrtf(t / (float) D + eps);
    }
    __syncthreads();
    const float inv = stat;
    for (int i = threadIdx.x; i < D; i += 256) out[(size_t) blockIdx.x * D + i] = (r[i] - mean) * inv * w[i];
}

__global__ void __launch_bounds__(1024) argmax_prob_kernel(const float* x, int n, int* best, float* prob) {
    __shared__ float bv[32];
    __shared__ int bi[32];
    __shared__ float tot[32];
    float v = -INFINITY;
    int id = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += 1024)
        if (better(x[i], i, v, id)) {
            v = x[i];
            id = i;
        }
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, v, o);
        const int oi = __shfl_xor_sync(0xffffffffu, id, o);
        if (better(ov, oi, v, id)) {
            v = ov;
            id = oi;
        }
    }
    if ((threadIdx.x & 31) == 0) {
        bv[threadIdx.x >> 5] = v;
        bi[threadIdx.x >> 5] = id;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        v = bv[threadIdx.x];
        id = bi[threadIdx.x];
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, v, o);
            const int oi = __shfl_xor_sync(0xffffffffu, id, o);
            if (better(ov, oi, v, id)) {
                v = ov;
                id = oi;
            }
        }
        if (threadIdx.x == 0) {
            bv[0] = v;
            bi[0] = id;
        }
    }
    __syncthreads();
    const float m = bv[0];
    float z = 0.0f;
    for (int i = threadIdx.x; i < n; i += 1024) z += expf(x[i] - m);
    for (int o = 16; o > 0; o >>= 1) z += __shfl_xor_sync(0xffffffffu, z, o);
    if ((threadIdx.x & 31) == 0) tot[threadIdx.x >> 5] = z;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.0f;
        for (int i = 0; i < 32; ++i) t += tot[i];
        *best = bi[0];
        *prob = 1.0f / t;
    }
}

}  // namespace

void layer_norm(const float* x, const float* w, int rows, int D, float eps, float* out, cudaStream_t s) {
    if (rows <= 0) return;
    layer_norm_kernel<<<rows, 256, 0, s>>>(x, w, D, eps, out);
    check("layer_norm");
}

void argmax_prob(const float* x, int n, int* best, float* prob, cudaStream_t s) {
    argmax_prob_kernel<<<1, 1024, 0, s>>>(x, n, best, prob);
    check("argmax_prob");
}

void conv_silu(const float* x, int ldx, int T, int C, const float* w, const float* bias, float* tail, float* out,
               cudaStream_t s) {
    if (T <= 0) return;
    conv_silu_kernel<<<(C + 127) / 128, 128, 0, s>>>(x, ldx, T, C, w, bias, tail, out);
    check("conv_silu");
}

void ssm_scan(const float* xbc, int T, int H, int P, int G, int N, const float* dt, int lddt, const float* dt_bias,
              const float* A, const float* D, float* state, float* y, cudaStream_t s) {
    if (T <= 0) return;
    if (P != kScanP || N != kScanN || H % G != 0) {
        std::fprintf(stderr, "nh ssm_scan: head_dim %d / state %d not covered (64 / 128)\n", P, N);
        return;
    }
    ssm_scan_kernel<<<H, 256, 0, s>>>(xbc, T, H, G, dt, lddt, dt_bias, A, D, state, y);
    check("ssm_scan");
}

void gated_norm(const float* y, const float* z, int ldz, int T, int n, int group, const float* w, float eps,
                float* out, uint16_t* out16, cudaStream_t s) {
    if (T <= 0) return;
    gated_norm_kernel<<<dim3((unsigned) T, (unsigned) (n / group)), 256, 0, s>>>(y, z, ldz, n, group, w, eps, out,
                                                                                out16);
    check("gated_norm");
}

void route(const float* logits, const float* bias, int T, int E, int K, float scale, int* top, float* w,
           cudaStream_t s) {
    if (T <= 0) return;
    if (E > 1024 || K > 32) {
        std::fprintf(stderr, "nh route: %d experts top %d not covered\n", E, K);
        return;
    }
    route_kernel<<<(T + 3) / 4, 128, 0, s>>>(logits, bias, T, E, K, scale, top, w);
    check("route");
}

void kv_store(const float* k, const float* v, const int* pos, int T, int Hkv, uint16_t* kc, uint16_t* vc,
              cudaStream_t s) {
    if (T <= 0) return;
    kv_store_kernel<<<T, Hkv * 128, 0, s>>>(k, v, pos, Hkv, kc, vc);
    check("kv_store");
}

void kept_all(const int* pos, int T, int Hkv, int nblk, int blk, int* kept, cudaStream_t s) {
    if (T <= 0) return;
    kept_all_kernel<<<dim3((unsigned) T, (unsigned) Hkv), 128, 0, s>>>(pos, T, Hkv, nblk, blk, kept);
    check("kept_all");
}

void relu2(const float* u, size_t n, float* hf, uint16_t* h16, cudaStream_t s) {
    if (n == 0) return;
    relu2_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, s>>>(u, n, hf, h16);
    check("relu2");
}

}  // namespace nh::k
