// include/strata/kernels/glm_dsv4.hpp - DeepSeek-V4 Flash's attention for the Glm5Model fast path (one token).
//
// A layer keeps a ring of the latest n_swa raw K/V rows (one 512-wide head: the value is the key vector), and - per
// its compress ratio - pooled rows: ratio 4 (CSA) one a 4 positions with a 128-wide indexer stream that picks top_k
// of them; ratio 128 (HCA) one a 128 positions, all visible.  The rows are FP16 (the non-rope 448 dims went through
// the model's FP8 E4M3 round trip, the indexer's through Hadamard + FP4 E2M1: both exact in FP16).  The reference is
// antirez's ds4.c (layer_forward_raw_swa_one and the functions it calls) and llama.cpp's deepseek4.cpp.
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace strata::kernels::glmf {

constexpr int kDsv4Hd = 512;    // head_dim (value = key)
constexpr int kDsv4Rot = 64;    // rope dims: the tail of a head
constexpr int kDsv4Ihd = 128;   // the indexer's head_dim

/// A layer's rope (rope_tail_layer_inplace): theta_i = pos * base^(-2 i / n_rot); compressed layers interpolate by
/// `scale` with YaRN's ramp over [corr0, corr1] (ext 1) and no magnitude change.
struct Dsv4Rope {
    float base = 10000.0f, scale = 1.0f, ext = 0.0f, corr0 = 0.0f, corr1 = 0.0f;
};

/// One layer's caches and compressor states (device memory; see Glm5Model's dsv4 state layout).
struct Dsv4Layer {
    uint16_t* swa = nullptr;     // n_swa x 512 FP16: position p at p % n_swa
    uint16_t* comp = nullptr;    // pooled rows x 512 FP16 (ratio > 0)
    uint16_t* icomp = nullptr;   // pooled indexer rows x 128 FP16 (ratio 4)
    float* st_kv = nullptr;      // the compressor's window: (ratio 4: 8, else ratio) rows x width (ratio 4: 1024, else 512)
    float* st_sc = nullptr;      //   ... their scores (-inf: empty)
    float* ist_kv = nullptr;     // the indexer compressor's (8 x 256)
    float* ist_sc = nullptr;
    int ratio = 0;
};

/// The per-token inputs of the cache update: the projections of the attention input.
struct Dsv4KvArgs {
    const float* kv_raw;         // attn_kv . x (512)
    const float* kv_norm;        // attn_kv_a_norm (512)
    const float* comp_kv;        // attn_compressor_kv . x (width)
    const float* comp_sc;        // attn_compressor_gate . x (width)
    const float* comp_ape;       // attn_compressor_ape [ratio][width]
    const float* comp_norm;      // attn_compressor_norm (512)
    const float* icomp_kv;       // indexer_compressor_kv . x (256)
    const float* icomp_sc;       // indexer_compressor_gate . x (256)
    const float* icomp_ape;      // indexer_compressor_ape [4][256]
    const float* icomp_norm;     // indexer_compressor_norm (128)
    float eps;
    int n_swa;
};

/// The raw row (norm, rope, FP8) into the ring and, when ratio > 0, the compressor states (+ a pooled row at every
/// ratio-th position, at index (pos + 1) / ratio - 1).  rope: the layer's.
void dsv4_kv_update(const Dsv4KvArgs& a, const Dsv4Layer& L, const Dsv4Rope& rope, int pos, cudaStream_t s);

/// The q LoRA's middle: qr_norm = rms_norm(qr) * w (1024) as f32 and q8_1.
void dsv4_qr_norm(const float* qr, const float* w, float eps, int n, float* out, void* out_q8, cudaStream_t s);

/// The indexer's pick over the layer's n_comp pooled rows (ratio 4, n_comp > top_k): q (n_head x 128, raw from
/// indexer.attn_q_b) gets rope + Hadamard + FP4, w (n_head, raw from indexer.proj) the 1/sqrt(128 n_head) scale;
/// score[c] = sum_h relu(q_h . k_c) w_h; sel <- the top_k rows (any order).  score: n_comp floats of scratch.
void dsv4_indexer(float* q, const float* w, int n_head, const Dsv4Layer& L, int n_comp, const Dsv4Rope& rope,
                  int pos, int top_k, float* score, int* sel, cudaStream_t s);

/// The attention of the 64 heads (q: raw from attn_q_b, n_head x 512 - its per-head RMS norm and rope are done
/// here) over the ring's n_raw rows and the pooled rows (sel: n_sel of them by index; nullptr: rows 0 .. n_sel - 1),
/// with the per-head sinks; the output's rope undone; out: n_head x 512 f32.
void dsv4_attn(const float* q, const float* sinks, const Dsv4Layer& L, int n_swa, int pos, const int* sel, int n_sel,
               const Dsv4Rope& rope, float eps, int n_head, float* out, cudaStream_t s);

/// The mHC head: pre = sigmoid(fn . rms_norm(R) * scale + base) + eps (4 weights), x = sum_h pre_h R_h, then
/// output_norm -> x (f32) and xq (q8_1).  fn FP16 [4][4 n_embd].
void dsv4_hc_head(const float* R, const uint16_t* fn, const float* scale, const float* base, float hc_eps,
                  const float* norm_w, float eps, int n_embd, float* x, void* xq, cudaStream_t s);

// ---- the prompt path: a sub-batch of T rows at positions p0 .. p0 + T - 1 (glm_prefill.cu)
/// qr = rms_norm(qr_raw) * w per row (n wide), F32 and FP16.
void dsv4_rows_qr(const float* qr_raw, const float* w, float eps, int T, int n, float* qr, uint16_t* qr16,
                  cudaStream_t s);
/// The raw K/V rows (norm, rope, FP8) -> raw16 (T x 512 FP16); the ring takes them after the attention.
void dsv4_rows_kv(const float* kv_raw, const float* w, float eps, int T, int p0, const Dsv4Rope& rope, uint16_t* raw16,
                  cudaStream_t s);
/// A compressor over the rows in order (the attention's, or with `indexer` the indexer's): kv / sc are its
/// projections (T x width), the pooled rows land in the layer's caches.
void dsv4_rows_comp(const float* kv, const float* sc, int T, int p0, const float* ape, const float* norm, float eps,
                    const Dsv4Layer& L, const Dsv4Rope& rope, bool indexer, cudaStream_t s);
/// The indexer's picks per row (q: T x n_head x 128 raw, w: T x n_head prescaled - both in place where noted): sel
/// (T x top_k) row t's top_k of its (p0 + t + 1) / 4 visible pooled rows, or all of them; score: T x ld scratch.
void dsv4_rows_indexer(float* q, const float* w, int n_head, const Dsv4Layer& L, int T, int p0, int top_k,
                       const Dsv4Rope& rope, float* score, int ld, int* sel, cudaStream_t s);
/// The attention per row (q raw, T x n_head x 512) over the raw window and the visible pooled rows (sel: the
/// indexer's, nullptr: all) -> out16 (T x n_head x 512 FP16).
void dsv4_rows_attn(const float* q, const float* sinks, const Dsv4Layer& L, const uint16_t* raw16, int n_swa, int p0,
                    int T, const int* sel, int k, const Dsv4Rope& rope, float eps, int n_head, uint16_t* out16,
                    cudaStream_t s);
/// The sub-batch's last n_swa raw rows into the layer's ring.
void dsv4_rows_ring(const uint16_t* raw16, const Dsv4Layer& L, int n_swa, int p0, int T, cudaStream_t s);

}  // namespace strata::kernels::glmf
