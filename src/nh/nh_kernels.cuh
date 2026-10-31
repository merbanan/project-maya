// src/nh/nh_kernels.cuh - maya-nh's own kernels around the EXL3 matrices: NVIDIA-Nemotron-3.5's (Nemotron-H) Mamba-2
// mixer (the causal depthwise conv, the selective scan, the gated RMSNorm), its sigmoid router over 512 experts, relu^2,
// and the attention's KV rows and kept blocks (maya-mm3's sparse attention, every block kept: exact causal attention).
// The arithmetic follows transformers' modeling_nemotron_h.py (tools/nh/ref.py) in F32; the KV cache is BF16.
//
// Layouts: rows of tokens.  The Mamba mixer's in_proj row is [z (H P) | xBC (H P + 2 G N) | dt (H)], its conv rows
// [x (H P) | B (G N) | C (G N)]; head h reads group h / (H / G).  A layer's Mamba state: conv [conv_dim][kConvTail] (the
// last inputs, oldest first), ssm [H][P][N] F32.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace nh::k {

constexpr int kConvTail = 3;   // (conv_kernel 4: the inputs before a chunk its first outputs read)

/// out [T][C] = silu(bias + sum_j w[c][j] x[t - 3 + j]) for the C channels of x (rows ldx floats apart), the inputs
/// before t = 0 from tail [C][3] (oldest first) - then tail = the last three inputs.
void conv_silu(const float* x, int ldx, int T, int C, const float* w /*[C][4]*/, const float* bias, float* tail,
               float* out, cudaStream_t s);

/// The selective scan of T tokens over the conv rows xbc [T][H P + 2 G N] and dt (rows lddt floats apart):
/// per head h, dt' = softplus(dt + dt_bias[h]), S = exp(dt' A[h]) S + dt' x B^T, y = S C + D[h] x -> y [T][H P];
/// state [H][P][N] carried.  A = -exp(A_log) (the caller's).  P = 64, N = 128.
void ssm_scan(const float* xbc, int T, int H, int P, int G, int N, const float* dt, int lddt, const float* dt_bias,
              const float* A, const float* D, float* state, float* y, cudaStream_t s);

/// The gated RMSNorm (norm_before_gate False): v = y * silu(z), rms over groups of `group` values, times w -> F32 out
/// (if given) and FP16 out16 (if given), rows of n = groups x group.  z: rows ldz floats apart.
void gated_norm(const float* y, const float* z, int ldz, int T, int n, int group, const float* w, float eps,
                float* out, uint16_t* out16, cudaStream_t s);

/// The router over logits [T][E] (F32), E <= 1024: s = sigmoid, chosen by s + bias (the K largest, ties to the lower
/// expert), weights s / (sum over the chosen + 1e-20) * scale -> top [T][K], w [T][K].  K <= 32.
void route(const float* logits, const float* bias, int T, int E, int K, float scale, int* top, float* w,
           cudaStream_t s);

/// K and V rows (F32 [T][Hkv][128]) into the BF16 caches at the tokens' positions.
void kv_store(const float* k, const float* v, const int* pos, int T, int Hkv, uint16_t* kc, uint16_t* vc,
              cudaStream_t s);

/// kept [Hkv][T][nblk]: every block up to the token's own (b <= pos[t] / blk), -1 after - maya-mm3's sparse attention
/// over them is exact causal attention.
void kept_all(const int* pos, int T, int Hkv, int nblk, int blk, int* kept, cudaStream_t s);

/// h = relu(u)^2 over n values: F32 into hf and / or FP16 into h16 (either may be null).
void relu2(const float* u, size_t n, float* hf, uint16_t* h16, cudaStream_t s);

}  // namespace nh::k
