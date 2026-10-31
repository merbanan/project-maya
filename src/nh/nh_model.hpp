// src/nh/nh_model.hpp - maya-nh: NVIDIA-Nemotron-3.5 (Nemotron-H, EXL3) on one GPU.
//
// The dense matrices (the Mamba mixers' in / out projections, attention, the MoE layers' latent projections, shared
// experts and routers, the head) in VRAM; then as many routed experts as fit (a blob each: up, down); the others in
// pinned host memory, copied into staging slots when a token routes to them.  The Mamba layers' state (conv tail, scan
// state) and the attention layers' BF16 KV cache are the sequence's: forward() runs tokens from the state's position
// on (pos0 == pos()), or restarts at 0; snapshot / restore keep the Mamba state at a position (the prefix a later
// prompt shares).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace nh {

struct Options {
    std::string pack;
    int gpu = 0;
    int max_ctx = 65536;
    int chunk = 1024;           // prompt tokens a forward takes at most
    int staging = 64;           // expert staging slots (each one expert's blob)
    double reserve_gb = 1.0;    // VRAM kept free beyond the plan
    int resident_cap = -1;      // routed experts kept in VRAM a layer at most (-1: as many as fit)
    std::string profile;        // an expert profile (uint32 [layers][experts] routings): the most used stay in VRAM
    bool pin_pool = true;       // the experts not in VRAM in pinned host memory (else assembled from the files a miss)
    int cpu_threads = -2;       // the CPU lane's workers besides the caller (-1: none - a decode's misses are staged;
                                // -2: this host's, CpuLane::auto_workers)
    bool mtp = false;           // load the MTP block (the pack's): speculative decoding with forward(verify) / commit
    bool verbose = false;
};

struct Stats {
    double load_s = 0;
    long staged = 0;            // expert blobs copied into staging slots
    long resident = 0, experts = 0;
    double vram_gb = 0;         // the resident experts' bytes
    double pinned_gb = 0;
    long profiled = 0;          // routings counted (save_profile writes them)
    long cpu_experts = 0;       // routed experts the CPU lane computed
    int cpu_threads = 0;        // the CPU lane's threads (0: no CPU lane)
};

class Model {
public:
    Model();
    ~Model();
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    bool load(const Options& o, std::string& err);
    /// Runs ids at positions pos0 .. pos0 + n - 1: pos0 must be pos() (the state's) or 0 (a new sequence) - in chunks
    /// of at most Options::chunk; logits_out (if given) gets the logits of the last `want` positions of this call,
    /// [want][vocab] F32.
    bool forward(const int* ids, int n, int pos0, int want, float* logits_out, std::string& err);
    /// A verify pass (Options::mtp): n <= 8 tokens at pos(), their logits (want of them); the position stays and the
    /// Mamba state is kept as it was until commit(k) keeps the first k (1 <= k <= n) - the KV rows stay by position.
    bool verify(const int* ids, int n, int want, float* logits_out, std::string& err);
    bool commit(int k, std::string& err);
    /// The MTP block (Options::mtp): the last forward's rows 0 .. n - 1 (positions p0 ..) into it - row j with its next
    /// token next[j] for j < known; a last row whose next token is not known yet (known = n - 1) waits for the anchor.
    bool has_mtp() const;
    bool mtp_inject(int n, int p0, const int* next, int known, std::string& err);
    /// Up to K greedy drafts after `anchor` (the token at pos()): stops after the draft whose running product of
    /// confidences (each its softmax probability) falls below p_cut (0: never); conf gets each draft's.
    bool mtp_draft(int anchor, int K, double p_cut, std::vector<int>& out, std::vector<float>& conf, std::string& err);
    /// The positions the state holds (the next token's).
    int pos() const;
    /// Keeps the Mamba state at pos() (the KV cache keeps its rows by position: nothing to copy) - restore() goes back
    /// to it, as long as no forward from 0 came between.
    bool snapshot(std::string& err);
    int snapshot_pos() const;   // (-1: none)
    bool restore(std::string& err);
    int vocab() const;
    int max_ctx() const;
    int bos() const;
    std::vector<int> eos() const;
    const Stats& stats() const;
    std::vector<std::string> info_lines() const;
    bool save_profile(const std::string& path, std::string& err) const;

private:
    struct Impl;
    Impl* p;
};

}  // namespace nh
