// include/strata/prefill/moe_tc_iq.hpp - a prompt's routed experts on Volta's FP16 tensor cores, IQ2_XXS / IQ2_S / IQ3_XXS
// weights (the GLM and Hy3 packs' experts).
//
// dst[r][c] = sum_k W_e[c][k] x[r][k] for the rows r in [bounds[e], bounds[e + 1]) of each expert e: W the experts'
// matrices expert_bytes apart (n_out rows of n_in values, GGUF blocks), x the rows as mmq::quantize writes them for that
// type ([n_in / 128][xq_rows] block_q8_1_mmq, D4), dst rows of ld floats.  The bounds may be offset: xq and dst are
// addressed with the bound values as rows (the caller shifts the pointers, as for MMQ).  FP16 products, FP32 sums.
// pair: the matrix is gate || up (n_out / 2 rows each) - the register-fragment kernel pairs row r with row n_out / 2 + r.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::prefill::tciq {

#if defined(STRATA_PREFILL_TCIQ)
bool supported(int type, int64_t n_in, int64_t n_out);
/// the device is sm_70 (on sm_75+ MMQ runs int8 tensor cores already)
bool available();
void product(int type, const void* W, size_t expert_bytes, int64_t n_in, int64_t n_out, int n_experts, const int32_t* bounds,
             int max_rows, const void* xq, int64_t xq_rows, float* dst, int64_t ld, void* stream, bool pair = false);
#else   // (not in this build: no-ops, so the caller needs no #ifdef)
inline bool supported(int, int64_t, int64_t) { return false; }
inline bool available() { return false; }
inline void product(int, const void*, size_t, int64_t, int64_t, int, const int32_t*, int, const void*, int64_t, float*, int64_t,
                    void*, bool = false) {}
#endif

}  // namespace strata::prefill::tciq
