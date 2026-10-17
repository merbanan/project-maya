// include/strata/kernels/mimo_attn.hpp - mimo2's attention (Xiaomi MiMo-V2.x-Flash, llama.cpp src/models/mimo2.cpp):
// grouped-query attention with K heads of 192 and V heads of 128, NeoX rope over the first n_rot dims of q and k, an
// optional per-head attention sink (an extra logit in the softmax with no value) and an optional sliding window (a
// query at pos attends to keys pos - window + 1 .. pos).  The K / V cache is FP16, position-major: K [slot][n_kv][192],
// V [slot][n_kv][128].
#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace strata::kernels::mimo {

constexpr int kDK = 192, kDV = 128;

/// A layer's K / V records: position pos is slot pos % cap (ring) or pos - pos0.
struct Kv {
    void* k = nullptr;   // __half [cap][n_kv][kDK]
    void* v = nullptr;   // __half [cap][n_kv][kDV]
    int n_kv = 0;
    int cap = 0;
    int64_t pos0 = 0;
    bool ring = false;
};

/// The fused projection's rows (q: n_head x 192, k: n_kv x 192, v: n_kv x 128, ld floats a row) for positions p0 ..
/// p0 + T - 1: q and k roped in place (NeoX over the first n_rot dims, theta = pos * base^(-2i / n_rot)), k and v
/// written into kv as FP16.
void prep(float* qkv, int64_t ld, int T, int64_t p0, int n_head, int n_rot, float base, const Kv& kv, cudaStream_t s);

/// Attention for the T query rows (q rows of qld floats, positions p0 ..): keys max(0, pos - window + 1) .. pos
/// (window 0: 0 .. pos), all present in kv; scores * scale; sinks (n_head logits) or null.  out: T rows of ldo
/// floats, head h's 128 values at h * 128, times out_scale (mimo2's value_scale).  scratch: attn_scratch_floats(n_head, T, max_keys)
/// floats for decode's split parts (T <= 16); null runs the prompt kernel, which needs none, for any T.
void attn(const float* q, int64_t qld, int T, int64_t p0, int n_head, const Kv& kv, int window, const float* sinks,
          float scale, float out_scale, float* out, int64_t ldo, float* scratch, cudaStream_t s);
size_t attn_scratch_floats(int n_head, int T, int64_t max_keys);

/// Positions [lo, hi) from one view into another (the sliding-window ring <-> a prompt chunk's staging copy).
void copy(const Kv& src, const Kv& dst, int64_t lo, int64_t hi, cudaStream_t s);

}  // namespace strata::kernels::mimo
