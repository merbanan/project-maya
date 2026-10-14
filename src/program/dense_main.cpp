// src/program/dense_main.cpp - `strata-dense --serve --native <model.gguf> --context N`
//
// The dense-model twin of `strata --serve`.  It speaks the SAME line protocol on stdin/stdout, so serve/server.py
// drives it through its unchanged StrataEngine (tokenizer, chat template, thinking levels, tools, OpenAI and
// Anthropic endpoints, web app, Monitor):
//
//   out   INFO key=value ...                      (optional facts for the Monitor)
//   out   READY <context> stop                    ("stop": this engine honours STOP)
//   in    GEN <max_new> [key=value ...] id,id,... (keys: temperature top_p top_k min_p penalty_last_n penalty_repeat
//                                                  penalty_freq penalty_present seed; unknown keys are skipped)
//   out   PP <position> <prompt tokens> <ms> <tok/s>          every few dozen prompt tokens
//   out   T <id>                                              one per generated token
//   out   DONE <generated> <prompt> <prompt ms> <decode ms> <stop|length|cancel> 0 0 <reused>
//   in    STOP | QUIT
//
// Conversation cache: the recurrent GDN state cannot be rolled back, so a request reuses the state only when
// EVERYTHING the state has consumed is a prefix of the new prompt (a chat that only appends).  Otherwise the
// state is reset and the prompt is read again.
//
// The prompt is read one token at a time (the same step as decoding).  Correct, and the simplest thing that can
// be checked against llama.cpp; a batched prefill is the next optimisation, not a prerequisite.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/dense_model.hpp"
#include "strata/artifact/dequant.hpp"
#include "strata/kernels/dense_kernels.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/kernels/q35_kernels.hpp"
#include "strata/spec/suffix_drafter.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <deque>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

struct Lines {
    std::mutex m;
    std::condition_variable cv;
    std::deque<std::string> q;
    std::atomic<bool> stop{false};
    bool eof = false;
    void push(std::string s) {
        std::lock_guard<std::mutex> l(m);
        q.push_back(std::move(s));
        cv.notify_one();
    }
    bool pop(std::string& out) {
        std::unique_lock<std::mutex> l(m);
        cv.wait(l, [&] { return !q.empty() || eof; });
        if (q.empty()) return false;
        out = std::move(q.front());
        q.pop_front();
        return true;
    }
};

bool parse_ids(const char* p, std::vector<int32_t>& ids) {
    ids.clear();
    while (*p == ' ') ++p;
    while (*p) {
        char* end = nullptr;
        const long long v = std::strtoll(p, &end, 10);
        if (end == p || v < 0 || v > INT32_MAX) return false;
        ids.push_back((int32_t) v);
        p = end;
        if (*p == ',') ++p;
        else if (*p != '\0' && *p != '\r' && *p != '\n') return false;
        else break;
    }
    return !ids.empty();
}

void usage() {
    std::fprintf(stderr,
                 "usage: strata-dense --serve --native <model.gguf> [--context N]\n"
                 "  a dense qwen35 GGUF (Qwen3.8-27B) behind Strata's server protocol; the server runs it with\n"
                 "  `serve/server.py --engine strata --config <run config>` whose `exe` is this program.\n");
}

// ---- `strata-dense --selftest`: the new CUDA kernels against a CPU reference, no model needed.
uint16_t f32_to_f16_bits(float f) {          // round to nearest even; the test inputs stay well inside the range
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t e = (int32_t) ((x >> 23) & 0xff) - 127 + 15;
    uint32_t m = x & 0x7fffffu;
    if (e <= 0) return (uint16_t) sign;     // flush subnormals: the test data never reaches them
    if (e >= 31) return (uint16_t) (sign | 0x7bffu);
    uint32_t h = sign | ((uint32_t) e << 10) | (m >> 13);
    const uint32_t rest = m & 0x1fffu;
    if (rest > 0x1000u || (rest == 0x1000u && (h & 1u))) ++h;
    return (uint16_t) h;
}

struct Diff { double abs = 0, rel = 0; };
Diff compare(const std::vector<float>& a, const std::vector<float>& ref) {
    Diff d;
    for (size_t i = 0; i < a.size(); ++i) {
        const double e = std::fabs((double) a[i] - (double) ref[i]);
        d.abs = std::max(d.abs, e);
        d.rel = std::max(d.rel, e / (std::fabs((double) ref[i]) + 1e-3));
    }
    return d;
}

template <class T> T* to_dev(const std::vector<T>& h) {
    T* d = nullptr;
    cudaMalloc((void**) &d, h.size() * sizeof(T));
    cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}
std::vector<float> from_dev(const float* d, size_t n) {
    std::vector<float> h(n);
    cudaMemcpy(h.data(), d, n * 4, cudaMemcpyDeviceToHost);
    return h;
}

int selftest() {
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    auto rnd = [&](size_t n, float sc = 1.0f) { std::vector<float> v(n); for (auto& x : v) x = nd(rng) * sc; return v; };
    int bad = 0;
    auto report = [&](const char* what, const Diff& d, double tol) {
        const bool ok = d.abs < tol;
        std::printf("%-34s max abs err %.3g  max rel %.3g  %s\n", what, d.abs, d.rel, ok ? "ok" : "FAIL");
        if (!ok) ++bad;
    };
    try {
        {   // RMSNorm
            const int rows = 3, cols = 5120;
            auto x = rnd((size_t) rows * cols, 3.0f), w = rnd(cols, 1.0f);
            std::vector<float> ref(x.size());
            for (int r = 0; r < rows; ++r) {
                double ss = 0; for (int i = 0; i < cols; ++i) ss += (double) x[(size_t) r * cols + i] * x[(size_t) r * cols + i];
                const double sc = 1.0 / std::sqrt(ss / cols + 1e-6);
                for (int i = 0; i < cols; ++i) ref[(size_t) r * cols + i] = (float) (x[(size_t) r * cols + i] * sc * w[i]);
            }
            float *dx = to_dev(x), *dw = to_dev(w), *dout = nullptr;
            cudaMalloc((void**) &dout, x.size() * 4);
            strata::kernels::dense_rms_norm(dx, dw, dout, rows, cols, 1e-6f, nullptr);
            report("rms_norm 3 x 5120", compare(from_dev(dout, x.size()), ref), 1e-4);
        }
        {   // SwiGLU
            const int n = 17408;
            auto g = rnd(n, 2.0f), u = rnd(n, 2.0f);
            std::vector<float> ref(n);
            for (int i = 0; i < n; ++i) ref[(size_t) i] = (float) (g[(size_t) i] / (1.0 + std::exp(-(double) g[(size_t) i])) * u[(size_t) i]);
            float *dg = to_dev(g), *du = to_dev(u), *dout = nullptr;
            cudaMalloc((void**) &dout, n * 4);
            strata::kernels::dense_swiglu(dg, du, dout, n, nullptr);
            report("swiglu 17408", compare(from_dev(dout, n), ref), 1e-4);
        }
        {   // F32 GEMV
            const int n_in = 5120, n_out = 48;
            auto W = rnd((size_t) n_in * n_out, 0.05f), x = rnd(n_in);
            std::vector<float> ref(n_out);
            for (int r = 0; r < n_out; ++r) { double a = 0; for (int i = 0; i < n_in; ++i) a += (double) W[(size_t) r * n_in + i] * x[(size_t) i]; ref[(size_t) r] = (float) a; }
            float *dW = to_dev(W), *dx = to_dev(x), *dy = nullptr;
            cudaMalloc((void**) &dy, n_out * 4);
            strata::kernels::dense_gemv_f32(dW, dx, dy, n_in, n_out, nullptr);
            report("gemv_f32 5120 -> 48", compare(from_dev(dy, n_out), ref), 1e-3);
        }
        for (int n_ctx : {1, 40, 256, 257, 700}) {   // attention, including several splits and a partial last one
            const int H = 24, HK = 4, D = 256, max_ctx = 1024;
            auto q = rnd((size_t) H * D), k = rnd((size_t) HK * max_ctx * D), v = rnd((size_t) HK * max_ctx * D);
            std::vector<uint16_t> kh(k.size()), vh(v.size());
            std::vector<float> kr(k.size()), vr(v.size());
            for (size_t i = 0; i < k.size(); ++i) {
                kh[i] = f32_to_f16_bits(k[i]); vh[i] = f32_to_f16_bits(v[i]);
                kr[i] = strata::fp16_to_fp32(kh[i]); vr[i] = strata::fp16_to_fp32(vh[i]);
            }
            const float scale = 1.0f / std::sqrt((float) D);
            std::vector<float> ref((size_t) H * D);
            for (int h = 0; h < H; ++h) {
                const int kvh = h / (H / HK);
                std::vector<double> sc((size_t) n_ctx);
                double mx = -1e300;
                for (int t = 0; t < n_ctx; ++t) {
                    double d = 0;
                    for (int i = 0; i < D; ++i) d += (double) q[(size_t) h * D + i] * kr[((size_t) kvh * max_ctx + t) * D + i];
                    sc[(size_t) t] = d * scale; mx = std::max(mx, sc[(size_t) t]);
                }
                double den = 0; for (auto& x : sc) { x = std::exp(x - mx); den += x; }
                for (int i = 0; i < D; ++i) {
                    double a = 0;
                    for (int t = 0; t < n_ctx; ++t) a += sc[(size_t) t] * vr[((size_t) kvh * max_ctx + t) * D + i];
                    ref[(size_t) h * D + i] = (float) (a / den);
                }
            }
            float* dq = to_dev(q);
            uint16_t *dk = to_dev(kh), *dv = to_dev(vh);
            float *dout = nullptr, *dscr = nullptr;
            cudaMalloc((void**) &dout, ref.size() * 4);
            cudaMalloc((void**) &dscr, strata::kernels::dense_attn_scratch_bytes(H, D, max_ctx));
            strata::kernels::dense_attn_decode(dq, dk, dv, dout, dscr, H, HK, D, n_ctx, max_ctx, scale, nullptr);
            char name[64];
            std::snprintf(name, sizeof name, "attention 24/4 x 256, n_ctx %d", n_ctx);
            report(name, compare(from_dev(dout, ref.size()), ref), 2e-3);
        }
        {   // KV append: the cell the attention will read
            const int HK = 4, D = 256, max_ctx = 64, pos = 9;
            auto k = rnd((size_t) HK * D), v = rnd((size_t) HK * D);
            std::vector<uint16_t> zero((size_t) HK * max_ctx * D, 0);
            uint16_t *dk = to_dev(zero), *dv = to_dev(zero);
            float *dkc = to_dev(k), *dvc = to_dev(v);
            strata::kernels::dense_kv_append(dk, dv, dkc, dvc, pos, HK, D, max_ctx, nullptr);
            std::vector<uint16_t> back(zero.size());
            cudaMemcpy(back.data(), dk, back.size() * 2, cudaMemcpyDeviceToHost);
            std::vector<float> got((size_t) HK * D), ref((size_t) HK * D);
            for (int h = 0; h < HK; ++h) for (int i = 0; i < D; ++i) {
                got[(size_t) h * D + i] = strata::fp16_to_fp32(back[((size_t) h * max_ctx + pos) * D + i]);
                ref[(size_t) h * D + i] = strata::fp16_to_fp32(f32_to_f16_bits(k[(size_t) h * D + i]));
            }
            report("kv_append (K cell)", compare(got, ref), 1e-6);
        }
    } catch (const std::exception& e) {
        std::printf("selftest: exception: %s\n", e.what());
        return 1;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) { std::printf("selftest: a kernel failed\n"); return 1; }
    std::printf(bad ? "selftest: %d FAILED\n" : "selftest: all passed\n", bad);
    return bad ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--selftest") return selftest();
    std::string gguf;
    long long context = 32768;
    bool serve = false;
    int draft = -1, chunk = 512;            // drafts a round (-1: STRATA_DENSE_DRAFT, else 3 / DFlash2 7; 0: off), prompt chunk
    std::string dflash;                     // a DFlash2 drafter GGUF (else STRATA_DENSE_DFLASH): it drafts, not the MTP block
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "strata-dense: %s needs a value\n", what); std::exit(2); }
            return argv[++i];
        };
        if (a == "--serve") serve = true;
        else if (a == "--native" || a == "--model" || a == "--gguf") gguf = next("--native");
        else if (a == "--context" || a == "-c") context = std::atoll(next("--context"));
        else if (a == "--draft" || a == "--mtp-draft") draft = std::atoi(next("--draft"));
        else if (a == "--chunk") chunk = std::atoi(next("--chunk"));
        else if (a == "--dflash") dflash = next("--dflash");
        else if (a == "--kv") setenv("STRATA_DENSE_KV", next("--kv"), 1);
        else if (a == "--help" || a == "-h") { usage(); return 0; }
        // the flags setup passes to `strata` (--gpu-layers, --cache, ...) mean nothing here and are not an error
    }
    if (!serve || gguf.empty()) { usage(); return 2; }

    strata::core::DenseModel model;
    std::string err;
    std::fprintf(stderr, "strata-dense: loading %s ...\n", gguf.c_str());
    if (const char* e = std::getenv("STRATA_DENSE_CHUNK")) chunk = std::atoi(e);
    if (!model.load(gguf, context, err, chunk)) {
        std::fprintf(stderr, "strata-dense: %s\n", err.c_str());
        std::printf("ERR %s\n", err.c_str());
        return 1;
    }
    const auto& cfg = model.config();
    if (dflash.empty() && std::getenv("STRATA_DENSE_DFLASH")) dflash = std::getenv("STRATA_DENSE_DFLASH");
    if (!dflash.empty() && !model.dflash_load(dflash, err)) {
        std::fprintf(stderr, "strata-dense: %s\n", err.c_str());
        std::printf("ERR %s\n", err.c_str());
        return 1;
    }
    const bool use_df = model.has_dflash();
    if (draft < 0) draft = std::getenv("STRATA_DENSE_DRAFT") ? std::atoi(std::getenv("STRATA_DENSE_DRAFT")) : (use_df ? 7 : 3);
    draft = std::max(0, std::min(draft, use_df ? model.dflash_max() : 7));
    if (!use_df && !model.has_mtp()) draft = 0;
    cudaStream_t cs = (cudaStream_t) model.stream();

    // the ids that end an answer, from the model's own vocabulary
    std::vector<int32_t> stop_ids;
    if (cfg.eos_id >= 0) stop_ids.push_back(cfg.eos_id);
    try {
        strata::GgufFile g(gguf, 1u << 20);
        if (const auto* toks = g.get("tokenizer.ggml.tokens"))
            for (size_t i = 0; i < toks->items.size(); ++i)
                if (toks->items[i].s == "<|im_end|>" || toks->items[i].s == "<|endoftext|>" ||   // Qwen
                    toks->items[i].s == "<|eot|>" ||                                                // Muse Glimmer's turn end
                    toks->items[i].s == "<turn|>")                                                  // Gemma 4's
                    stop_ids.push_back((int32_t) i);
    } catch (const std::exception&) {}
    int32_t im_start = -1;   // the chat template's message header: prompts are checkpointed at its last occurrence
    try {
        strata::GgufFile g(gguf, 1u << 20);
        if (const auto* toks = g.get("tokenizer.ggml.tokens"))
            for (size_t i = 0; i < toks->items.size(); ++i)
                if (toks->items[i].s == "<|im_start|>" || toks->items[i].s == "<|start|>" || toks->items[i].s == "<|turn>") {
                    im_start = (int32_t) i;   // (Qwen's, Muse Glimmer's, Gemma 4's message header)
                    break;
                }
    } catch (const std::exception&) {}
    std::sort(stop_ids.begin(), stop_ids.end());
    stop_ids.erase(std::unique(stop_ids.begin(), stop_ids.end()), stop_ids.end());

    int* d_next = nullptr;
    int* d_hist = nullptr;
    float* d_hrows = nullptr;                      // the trunk's hidden rows the MTP block has yet to read
    constexpr int kMaxHist = 4096;
    if (cudaMalloc((void**) &d_next, 16 * sizeof(int)) != cudaSuccess ||
        cudaMalloc((void**) &d_hist, kMaxHist * sizeof(int)) != cudaSuccess ||
        cudaMalloc((void**) &d_hrows, (size_t) 16 * cfg.n_embd * sizeof(float)) != cudaSuccess) {
        std::printf("ERR cannot allocate the sampler buffers\n");
        return 1;
    }

    std::printf("INFO engine=dense-0.2 kv=%s model=%s layers=%d n_embd=%d weights_mib=%llu host_mib=%llu kv_mib=%llu context=%lld mtp=%d\n",
                std::getenv("STRATA_DENSE_KV") ? std::getenv("STRATA_DENSE_KV") : "fp16",
                cfg.arch.c_str(), cfg.n_layer, cfg.n_embd, (unsigned long long) (model.weight_bytes() >> 20),
                (unsigned long long) (model.host_weight_bytes() >> 20), (unsigned long long) (model.kv_bytes() >> 20),
                (long long) model.max_context(), draft);
    std::printf("READY %lld stop\n", (long long) model.max_context());
    std::fflush(stdout);
    std::fprintf(stderr, "strata-dense: ready - %.2f GiB of weights, %.2f GiB of KV cache, context %lld, MTP drafts %d\n",
                 model.weight_bytes() / 1073741824.0, model.kv_bytes() / 1073741824.0, (long long) model.max_context(),
                 draft);

    if (const char* e = std::getenv("STRATA_DENSE_BENCH")) {
        // the cost of a forward pass by width: a prompt of N tokens, then verify windows of 1..8 (each committed 1)
        const int n0 = std::max(16, std::atoi(e));
        std::vector<int32_t> toks((size_t) n0 + 64, 1000);
        for (size_t i = 0; i < toks.size(); ++i) toks[i] = (int32_t) (1000 + (i * 7919) % 50000);
        for (int i = 0; i < n0; i += model.chunk()) {
            const int T = std::min(model.chunk(), n0 - i);
            if (!model.forward(toks.data() + i, T, false, 0, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
        }
        model.sync(err);
        const int wmax = std::getenv("STRATA_DENSE_BENCH_W") ? std::atoi(std::getenv("STRATA_DENSE_BENCH_W")) : 8;
        const int reps = std::getenv("STRATA_DENSE_BENCH_REPS") ? std::atoi(std::getenv("STRATA_DENSE_BENCH_REPS")) : 12;
        for (int T = std::getenv("STRATA_DENSE_BENCH_W") ? wmax : 1; T <= wmax; ++T) {
            double tot = 0;
            for (int r = 0; r < reps; ++r) {
                const auto t0 = Clock::now();
                if (!model.forward(toks.data() + n0, T, true, strata::core::DenseModel::kAll, err) || !model.sync(err)) {
                    std::fprintf(stderr, "%s\n", err.c_str()); return 1;
                }
                if (r >= 2) tot += ms_since(t0);
                model.commit(1, err);
            }
            std::fprintf(stderr, "bench ctx %lld: verify window %d: %.2f ms\n", (long long) model.position(), T, tot / (reps - 2));
        }
        if (std::getenv("STRATA_DENSE_BENCH_GRAPH")) {   // the same pass captured once and replayed: launch overhead
            const int T = wmax;
            cudaStream_t gs = (cudaStream_t) model.stream();
            cudaGraph_t graph;
            cudaGraphExec_t exec;
            cudaStreamBeginCapture(gs, cudaStreamCaptureModeGlobal);
            const bool ok = model.forward(toks.data() + n0, T, true, strata::core::DenseModel::kAll, err);
            cudaStreamEndCapture(gs, &graph);
            model.commit(1, err);
            size_t nodes = 0;
            cudaGraphGetNodes(graph, nullptr, &nodes);
            if (!ok || cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) != cudaSuccess) {
                std::fprintf(stderr, "bench graph: capture failed: %s %s\n", err.c_str(), cudaGetErrorString(cudaGetLastError()));
                return 1;
            }
            double tot = 0;
            for (int r = 0; r < reps; ++r) {
                const auto t0 = Clock::now();
                cudaGraphLaunch(exec, gs);
                cudaStreamSynchronize(gs);
                if (r >= 2) tot += ms_since(t0);
            }
            std::fprintf(stderr, "bench graph: verify window %d replayed: %.2f ms (%zu nodes)\n", T, tot / (reps - 2), nodes);
        }
        return 0;
    }

    Lines in;
    std::thread reader([&] {
        std::string line;
        while (std::getline(std::cin, line)) {
            while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
            if (line == "STOP") { in.stop.store(true); continue; }
            in.push(line);
        }
        std::lock_guard<std::mutex> l(in.m);
        in.eof = true;
        in.cv.notify_all();
    });
    reader.detach();

    // context lookup (NInfer's, Strata's suffix drafter): when the text's last tokens occurred before - code being
    // edited, a tool's output quoted - the tokens that followed are proposed instead of the learned drafter's
    const int lookup_min = std::getenv("STRATA_DENSE_LOOKUP_MIN") ? std::atoi(std::getenv("STRATA_DENSE_LOOKUP_MIN")) : 12;
    strata::spec::SuffixDrafter lookup(std::max(3, lookup_min), 64);
    // prefix checkpoints (FreeToken's anchors, lithos-metal's prefix checkpoints): the recurrent state at the start of
    // a prompt's last message header, in pinned host memory; a later prompt that shares that prefix but not the rest
    // of the history (an agent dropped the reasoning, rewrote a tool call) restores it instead of reading from 0
    const int n_ckpt = std::getenv("STRATA_DENSE_CKPTS") ? std::max(0, std::atoi(std::getenv("STRATA_DENSE_CKPTS"))) : 8;
    struct Ckpt { int64_t len = -1; uint64_t hash = 0, used = 0; };
    std::vector<Ckpt> ckpts((size_t) n_ckpt);
    uint64_t ckpt_clock = 0;
    auto prefix_hash = [](const std::vector<int32_t>& v, int64_t n) {
        uint64_t h = 1469598103934665603ull;
        for (int64_t i = 0; i < n; ++i) { h ^= (uint32_t) v[(size_t) i]; h *= 1099511628211ull; }
        return h;
    };
    if (n_ckpt > 0)
        std::fprintf(stderr, "strata-dense: %d prefix checkpoints of %.0f MiB (pinned host memory, on first use), "
                     "at the last header %d\n", n_ckpt, model.ckpt_bytes() / 1048576.0, im_start);
    std::vector<int32_t> fed;                      // every token the state has consumed, in order
    uint64_t draw_counter = 0;
    std::string line;
    while (in.pop(line)) {
        if (line == "QUIT") break;
        in.stop.store(false);                       // a STOP between requests is stale
        if (line.rfind("GEN ", 0) != 0) { std::printf("ERR expected: GEN <max_new> [key=value ...] <id,id,...>\n"); std::fflush(stdout); continue; }

        char* endp = nullptr;
        const long long max_new = std::strtoll(line.c_str() + 4, &endp, 10);
        strata::kernels::SamplerParams sp;
        sp.top_k = 20; sp.top_p = 1.0f; sp.temperature = 0.0f; sp.greedy = true;
        unsigned long long seed = 0;
        bool ignore_eos = false;
        int req_draft = draft;
        long long think_budget = 0;               // think_budget=<n> think_end=<id>: the server's reasoning budget
        int32_t think_end = -1;
        for (;;) {
            while (*endp == ' ') ++endp;
            const char* start = endp;
            while (*endp != '\0' && *endp != ' ') ++endp;
            if (endp == start) break;
            const std::string tok(start, (size_t) (endp - start));
            const size_t eq = tok.find('=');
            if (eq == std::string::npos) { endp = const_cast<char*>(start); break; }
            const std::string key = tok.substr(0, eq);
            const float fv = std::strtof(tok.c_str() + eq + 1, nullptr);
            if (key == "temperature") { sp.temperature = fv; sp.greedy = fv <= 0.0f; }
            else if (key == "top_p") sp.top_p = fv;
            else if (key == "top_k") sp.top_k = std::atoi(tok.c_str() + eq + 1);
            else if (key == "min_p") sp.min_p = fv;
            else if (key == "penalty_last_n") sp.penalty_last_n = std::min(std::atoi(tok.c_str() + eq + 1), kMaxHist);
            else if (key == "penalty_repeat") sp.penalty_repeat = fv;
            else if (key == "penalty_freq") sp.penalty_freq = fv;
            else if (key == "penalty_present") sp.penalty_present = fv;
            else if (key == "seed") seed = std::strtoull(tok.c_str() + eq + 1, nullptr, 10);
            else if (key == "ignore_eos") ignore_eos = fv != 0.0f;
            else if (key == "think_budget") think_budget = std::atoll(tok.c_str() + eq + 1);
            else if (key == "think_end") think_end = (int32_t) std::atol(tok.c_str() + eq + 1);
            else if (key == "draft") req_draft = std::max(0, std::min(draft, std::atoi(tok.c_str() + eq + 1)));
        }
        std::vector<int32_t> ids;
        if (max_new < 1 || !parse_ids(endp, ids)) { std::printf("ERR bad request\n"); std::fflush(stdout); continue; }
        if ((long long) ids.size() + 1 > model.max_context()) {
            std::printf("ERR the prompt (%zu tokens) does not fit the context (%lld)\n", ids.size(), (long long) model.max_context());
            std::fflush(stdout);
            continue;
        }
        sp.seed = seed;
        auto is_stop = [&](int32_t t) { return !ignore_eos && std::binary_search(stop_ids.begin(), stop_ids.end(), t); };
        // penalties look at a history the verify rows would each need their own of: those requests decode plainly
        const bool penalties = sp.penalty_last_n > 0 &&
                               (sp.penalty_repeat != 1.0f || sp.penalty_freq != 0.0f || sp.penalty_present != 0.0f);
        const int K = penalties ? 0 : req_draft;

        // ---- what of the state can be kept (the MTP cache follows the trunk: the same positions are valid)
        size_t resume = 0;
        if (!fed.empty() && fed.size() < ids.size() && std::equal(fed.begin(), fed.end(), ids.begin())) resume = fed.size();
        bool restored = false;
        if (resume == 0 && !fed.empty() && n_ckpt > 0) {
            // the K/V cache holds fed's tokens: a checkpoint inside the common prefix of fed and the prompt is valid
            size_t cp = 0;
            while (cp < fed.size() && cp + 1 < ids.size() && fed[cp] == ids[cp]) ++cp;
            int best = -1;
            for (int c = 0; c < n_ckpt; ++c)
                if (ckpts[(size_t) c].len > 0 && (size_t) ckpts[(size_t) c].len <= cp &&
                    (best < 0 || ckpts[(size_t) c].len > ckpts[(size_t) best].len) &&
                    ckpts[(size_t) c].hash == prefix_hash(ids, ckpts[(size_t) c].len))
                    best = c;
            std::fprintf(stderr, "strata-dense: history diverges at %zu of %zu; checkpoint %d\n", cp, fed.size(), best);
            if (best >= 0 && model.ckpt_restore(best, ckpts[(size_t) best].len, err)) {
                resume = (size_t) ckpts[(size_t) best].len;
                fed.resize(resume);
                ckpts[(size_t) best].used = ++ckpt_clock;
                restored = true;
            }
        }
        if (resume == 0) { model.reset(); fed.clear(); }
        if (resume == 0 || lookup.size() != resume) {   // (the lookup history is the prompt and the answer so far)
            lookup.reset();
            lookup.append(ids.data(), ids.size());
        } else {
            lookup.append(ids.data() + resume, ids.size() - resume);
        }
        const int64_t n = (int64_t) ids.size();
        const long long budget = std::min<long long>(max_new, model.max_context() - n);

        // ---- read the prompt, a chunk at a time; the MTP block reads every row whose next token is known
        bool cancelled = false, failed = false;
        const auto t_prompt = Clock::now();
        const int ch = model.chunk();
        int64_t anchor_at = -1;   // the last message header (a checkpoint), when it is still to be read
        if (n_ckpt > 0 && im_start >= 0)
            for (int64_t j = n - 1; j > (int64_t) resume; --j)
                if (ids[(size_t) j] == im_start) { anchor_at = j; break; }
        for (int64_t i = (int64_t) resume; i < n && !failed;) {
            int T = (int) std::min<int64_t>(ch, n - i);
            if (anchor_at > i && anchor_at < i + T) T = (int) (anchor_at - i);   // a chunk ends at the header
            const bool last = i + T == n;
            if (!model.forward(ids.data() + i, T, false, last ? strata::core::DenseModel::kLast : strata::core::DenseModel::kNone, err)) {
                std::printf("ERR %s\n", err.c_str()); failed = true; break;
            }
            if (K > 0 && use_df) {
                if (!model.dflash_inject(T, i, err)) { std::printf("ERR %s\n", err.c_str()); failed = true; break; }
            } else if (K > 0) {
                const int rows = last ? T - 1 : T;   // row t pairs with token i + t + 1
                if (rows > 0 && !model.mtp_forward(ids.data() + i + 1, model.hidden_norm(), i, rows, false, err)) {
                    std::printf("ERR %s\n", err.c_str()); failed = true; break;
                }
                if (last)   // the prompt's last hidden row: the first draft step reads it
                    cudaMemcpyAsync(d_hrows, model.hidden_norm() + (size_t) (T - 1) * cfg.n_embd, cfg.n_embd * 4,
                                    cudaMemcpyDeviceToDevice, cs);
            }
            fed.insert(fed.end(), ids.begin() + i, ids.begin() + i + T);
            i += T;
            if (i == anchor_at) {
                const uint64_t h = prefix_hash(ids, i);
                int slot = -1;
                for (int c = 0; c < n_ckpt && slot < 0; ++c)
                    if (ckpts[(size_t) c].len == i && ckpts[(size_t) c].hash == h) slot = -2;   // (already there)
                if (slot == -1) {
                    slot = 0;
                    for (int c = 1; c < n_ckpt; ++c)
                        if (ckpts[(size_t) c].used < ckpts[(size_t) slot].used) slot = c;
                    if (model.ckpt_save(slot, err)) {
                        ckpts[(size_t) slot] = {i, h, ++ckpt_clock};
                        std::fprintf(stderr, "strata-dense: checkpoint %d at position %lld\n", slot, (long long) i);
                    }
                    else { std::fprintf(stderr, "strata-dense: %s\n", err.c_str()); ckpts[(size_t) slot].len = -1; }
                }
            }
            if (!model.sync(err)) { std::printf("ERR %s\n", err.c_str()); failed = true; break; }
            const double ms = ms_since(t_prompt);
            std::printf("PP %lld %lld %.0f %.1f\n", (long long) i, (long long) n, ms,
                        ms > 0 ? 1000.0 * (i - (int64_t) resume) / ms : 0.0);
            std::fflush(stdout);
            if (in.stop.load()) { cancelled = true; break; }
        }
        if (failed) { model.reset(); fed.clear(); std::fflush(stdout); continue; }
        const double prompt_ms = ms_since(t_prompt);

        // ---- decode
        long long produced = 0;
        long long rounds = 0, drafted = 0, accepted = 0;
        const char* finish = "length";
        const auto t_dec = Clock::now();
        std::vector<int32_t> hist;                  // penalty window, most recent last
        if (sp.penalty_last_n > 0) {
            const size_t take = std::min<size_t>((size_t) sp.penalty_last_n, fed.size());
            hist.assign(fed.end() - (std::ptrdiff_t) take, fed.end());
        }
        // the reasoning budget: past it, the next token not yet fed (a round's last pick, the next anchor) becomes the
        // end of the reasoning - as the GLM engine forces it
        bool in_think = think_budget > 0 && think_end >= 0;
        auto budget_force = [&](int32_t& t) {
            if (in_think && produced >= think_budget && t != think_end) t = think_end;
        };
        long long lookup_rounds = 0;
        auto emit = [&](int32_t t) -> bool {        // false: the answer ends here
            if (in_think && t == think_end) in_think = false;
            lookup.append(t);
            ++produced;
            std::printf("T %d\n", t);
            if (sp.penalty_last_n > 0) { hist.push_back(t); if ((int) hist.size() > sp.penalty_last_n) hist.erase(hist.begin()); }
            if (is_stop(t)) { finish = "stop"; return false; }
            return produced < budget;
        };
        auto sample_rows = [&](const float* logits, int rows, int* out_host) -> bool {
            sp.counter = draw_counter;
            const int* hp = nullptr;
            int hl = 0;
            if (sp.penalty_last_n > 0) {
                std::vector<int32_t> row((size_t) sp.penalty_last_n, -1);
                std::copy(hist.begin(), hist.end(), row.end() - (std::ptrdiff_t) hist.size());
                cudaMemcpyAsync(d_hist, row.data(), row.size() * sizeof(int), cudaMemcpyHostToDevice, cs);
                hp = d_hist;
                hl = sp.penalty_last_n;
            }
            try {
                if (sp.greedy) strata::kernels::q35_argmax_rows(logits, cfg.n_vocab, cfg.n_vocab, rows, d_next, cs);
                else strata::kernels::sample_tokens(logits, rows, cfg.n_vocab, hp, hl, sp, d_next, cs);
            } catch (const std::exception& e) { std::printf("ERR sampler: %s\n", e.what()); return false; }
            cudaMemcpyAsync(out_host, d_next, rows * sizeof(int), cudaMemcpyDeviceToHost, cs);
            if (!model.sync(err)) { std::printf("ERR %s\n", err.c_str()); return false; }
            draw_counter += rows;
            return true;
        };
        int anchor = -1;
        double draft_ms = 0;
        if (!cancelled && budget > 0) {
            if (!sample_rows(model.logits(), 1, &anchor)) failed = true;
            else { budget_force(anchor); emit(anchor); }                         // (a stop token sets finish)
        }
        bool going = !failed && !cancelled && budget > 0 && produced < budget && finish != std::string("stop");
        // the MTP rows still to read: tokens mtp_tok at positions mtp_p0 .. with the trunk's hidden rows in d_hrows
        std::vector<int32_t> mtp_tok = {anchor};
        int64_t mtp_p0 = model.position() - 1;
        while (going) {
            if (in.stop.load()) { cancelled = true; break; }
            const int64_t pos = model.position();
            int k = (int) std::min<long long>(K, std::min<long long>(budget - produced, model.max_context() - pos - 1));
            k = std::max(0, k);
            std::vector<int32_t> win = {anchor};
            int32_t lk[16];
            const int nl = k > 0 && lookup_min > 0 ? lookup.propose(k, lk) : 0;
            if (nl > 0 && lookup.last_match() >= lookup_min) {
                win.insert(win.end(), lk, lk + nl);
                ++lookup_rounds;
                // the MTP block still reads its pending rows (its cache follows the trunk)
                if (!use_df && !model.mtp_forward(mtp_tok.data(), d_hrows, mtp_p0, (int) mtp_tok.size(), false, err)) {
                    std::printf("ERR %s\n", err.c_str()); failed = true; break;
                }
            } else if (k > 0 && use_df) {
                std::vector<int32_t> d;
                const auto t_d = Clock::now();
                if (!model.dflash_draft(anchor, k, d, err)) { std::printf("ERR %s\n", err.c_str()); failed = true; break; }
                draft_ms += ms_since(t_d);
                win.insert(win.end(), d.begin(), d.end());
            } else if (k > 0) {
                // the drafts: the MTP block over the pending rows (the last one's logits), then one row a draft
                if (!model.mtp_forward(mtp_tok.data(), d_hrows, mtp_p0, (int) mtp_tok.size(), false, err)) {
                    std::printf("ERR %s\n", err.c_str()); failed = true; break;
                }
                for (int j = 0; j < k; ++j) {
                    int d = 0;
                    strata::kernels::q35_argmax_rows(model.mtp_logits(), model.draft_vocab(), model.draft_vocab(), 1, d_next, cs);
                    cudaMemcpyAsync(&d, d_next, sizeof(int), cudaMemcpyDeviceToHost, cs);
                    if (!model.sync(err)) { std::printf("ERR %s\n", err.c_str()); failed = true; break; }
                    win.push_back(d);
                    if (j + 1 < k && !model.mtp_forward(&d, model.mtp_hidden(), pos + j, 1, false, err)) {
                        std::printf("ERR %s\n", err.c_str()); failed = true; break;
                    }
                }
                if (failed) break;
            }
            const int W = (int) win.size();
            if (!model.forward(win.data(), W, K > 0, K > 0 ? strata::core::DenseModel::kAll : strata::core::DenseModel::kLast, err)) {
                std::printf("ERR %s\n", err.c_str()); failed = true; break;
            }
            int picks[16];
            if (!sample_rows(model.logits(), W, picks)) { failed = true; break; }
            int acc = 0;
            while (acc + 1 < W && win[acc + 1] == picks[acc]) ++acc;
            if (in_think) {   // the budget runs out inside the kept drafts: keep them up to it, then force the end
                const long long room = std::max(0LL, think_budget - produced);
                for (int j = 0; j < acc; ++j)
                    if (win[j + 1] == think_end) { acc = j + 1; break; }
                if (acc > room) { acc = (int) room; }
                const long long before = produced;
                produced += acc;   // (budget_force reads the count after the kept drafts)
                budget_force(picks[acc]);
                produced = before;
            }
            ++rounds; drafted += W - 1; accepted += acc;
            if (K > 0 && use_df) {
                if (!model.commit(acc + 1, err) || !model.dflash_inject(acc + 1, pos, err)) {
                    std::printf("ERR %s\n", err.c_str()); failed = true; break;
                }
            } else if (K > 0) {
                if (!model.commit(acc + 1, err)) { std::printf("ERR %s\n", err.c_str()); failed = true; break; }
                // the MTP block's next rows: the kept window's tokens after the anchor and the new anchor, with the
                // trunk's hidden rows of the anchor .. the last kept draft
                cudaMemcpyAsync(d_hrows, model.hidden_norm(), (size_t) (acc + 1) * cfg.n_embd * 4, cudaMemcpyDeviceToDevice, cs);
                mtp_tok.assign(win.begin() + 1, win.begin() + 1 + acc);
                mtp_tok.push_back(picks[acc]);
                mtp_p0 = pos;
            }
            fed.insert(fed.end(), win.begin(), win.begin() + acc + 1);
            bool more = true;
            for (int j = 0; j < acc && more; ++j) more = emit(win[j + 1]);
            if (more) more = emit(picks[acc]);
            std::fflush(stdout);
            anchor = picks[acc];
            going = more;
        }
        if (failed) { model.reset(); fed.clear(); std::fflush(stdout); continue; }
        if (cancelled) finish = "cancel";
        const double decode_ms = ms_since(t_dec);
        std::printf("DONE %lld %lld %.1f %.1f %s %lld %lld %lld\n", produced, (long long) n, prompt_ms, decode_ms, finish,
                    accepted, drafted, (long long) resume);
        std::fflush(stdout);
        std::fprintf(stderr, "strata-dense: prompt %lld tokens = %zu reused%s + %lld read in %.0f ms (%.0f tok/s), %lld generated "
                     "in %.0f ms (%.1f tok/s)%s; %s %lld rounds (%lld by lookup), %lld of %lld drafts kept (%.2f tokens a round, drafter %.1f ms a draft)\n",
                     (long long) n, resume, restored ? " (checkpoint)" : "", (long long) (n - (long long) resume), prompt_ms,
                     prompt_ms > 0 ? 1000.0 * (n - (long long) resume) / prompt_ms : 0.0, produced, decode_ms,
                     decode_ms > 0 ? 1000.0 * produced / decode_ms : 0.0, cancelled ? " (cancelled)" : "",
                     use_df ? model.dflash_kind() : "MTP", rounds, lookup_rounds, accepted, drafted, rounds ? (double) (accepted + rounds) / rounds : 0.0,
                     rounds > lookup_rounds ? draft_ms / (rounds - lookup_rounds) : 0.0);
    }
    cudaFree(d_next);
    cudaFree(d_hist);
    return 0;
}
