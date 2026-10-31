// src/core/glm_kv_stream.hpp - hy_v3's KV streaming state on one part (Glm5Model::kvs_; STRATA_GLM_KV_STREAM).
//
// The whole K/V cache in pinned, device-mapped host memory - per streamed layer its records [max_ctx][slice] - and in
// VRAM per layer a cache of R pages (glm_fast.hpp GqaStream: slots, page table, CLOCK state, page summaries), one
// decode step's selection buffers shared by the layers, and a staging copy of one layer's records (the prompt path's
// dense attention reads the earlier context from it).
#pragma once

#include "strata/kernels/glm_fast.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::core {

struct GqaKvStream {
    std::vector<strata::kernels::glmf::GqaStream> lay;   // by layer index (n_layers + 1: the draft block last)
    std::vector<int> layers;                             // the streamed layers (the draft block's first)
    strata::kernels::glmf::GqaSelBufs sel;
    uint8_t* host = nullptr;    // pinned (cudaHostAlloc, mapped)
    size_t host_bytes = 0;
    uint8_t* dev = nullptr;     // the slots, tables, summaries, selection buffers: one allocation
    uint8_t* stage = nullptr;   // [max_ctx][slice]
    size_t slice = 0;           // one layer's K and V rows of a position
    int budget = 0, sink = 1, recent = 0;   // pages a decode step attends to: by score, the first, the last
    // STRATA_GLM_KV_RESELECT=<N> (64; 1: every position): a layer keeps its selection for up to N decode positions within one page (its
    // slots change only in its own resolve, and the page being written is cached: the new row lands in its copy) -
    // then each layer's selection in lsel / lnsel, the position it was made at in sel_pos (-1: none; a prompt pass or
    // a reload drops them all)
    int reselect = 64;
    int* lsel = nullptr;        // [n_layers + 1][sel.max_sel]
    int* lnsel = nullptr;       // [n_layers + 1]
    std::vector<int64_t> sel_pos;
    void forget() { std::fill(sel_pos.begin(), sel_pos.end(), (int64_t) -1); }
};

}  // namespace strata::core
