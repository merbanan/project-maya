// src/core/dense_model.cpp - see include/strata/core/dense_model.hpp.
#include "strata/core/dense_model.hpp"

#include "strata/artifact/dequant.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/dense_kernels.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/q35_dflash.hpp"
#include "strata/kernels/q35_kernels.hpp"
#include "strata/prefill/gemm.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <exception>
#include <map>
#include <regex>
#include <set>
#include <thread>

namespace strata::core {
namespace {

using namespace strata::kernels;

constexpr uint32_t kF32 = 0, kF16 = 1, kBF16 = 30;
constexpr int kMaxCols = 8;                         // native_mmvq's columns; more tokens take the GEMM path

// The GGUF types a quantized weight may have: exactly what native_mmvq takes.
const char* kSupported = "Q4_0, Q5_0, Q8_0, Q2_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS";

// defined below: dequantizes contiguous blocks of any type the host knows (F32/F16/BF16 and the quantized ones)
bool dequant_row(uint32_t type, const uint8_t* src, int n, float* out);

struct QMat {
    void* dev = nullptr;
    int type = -1;
    int n_in = 0, n_out = 0;
    bool packed = false;                            // in q35_mmtp's packed layout (else GGUF rows)
};

struct Layer {
    bool attn = false;
    float* attn_norm = nullptr;
    float* post_norm = nullptr;
    QMat ffn_gate, ffn_up, ffn_down;
    // GDN mixer
    QMat qkv, z, ssm_out;
    float *conv_w = nullptr, *dt = nullptr, *ssm_a = nullptr, *ssm_norm = nullptr;
    float *ab_w = nullptr;                         // F32/F16/BF16 files: [alpha rows | beta rows] (2 h_v x n_embd)
    QMat alpha_q, beta_q;                          // quantized files
    bool ab_quant = false;
    float *state = nullptr, *conv_state = nullptr;
    // a verify pass's inputs of the recurrence (kMaxCols tokens): raw conv input, conv output (q, k normed), g, b
    float *st_raw = nullptr, *st_conv = nullptr, *st_ab = nullptr;   // st_ab rows: [g (h_v) | b (h_v)]
    // attention mixer
    QMat q, k, v, o;
    float *q_norm = nullptr, *k_norm = nullptr;
    Q35Kv kv;                                      // the K / V cache (FP16 or K8V4)
    // the MTP (nextn) block's own
    QMat eh_proj;
    float *enorm = nullptr, *hnorm = nullptr, *head_norm = nullptr;
};

struct DfLayer {
    float *attn_norm = nullptr, *ffn_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr;
    float *attn_base = nullptr, *ffn_base = nullptr;   // [side][tap][E]
    QMat q, k, v, o, attn_conv, ffn_conv, gate, up, down;
    uint16_t *Kr = nullptr, *Vr = nullptr;            // the context ring, [ring][n_kv * 128] FP16
};

struct Dflash {
    bool on = false;
    int n_layer = 0, n_head = 0, n_kv = 0, block = 0, taps = 0, group = 0, rank = 0, top_k = 0, window = 2048;
    int mask_tok = 0, n_cand = 0;                     // candidates come from ids [0, n_cand)
    float eps = 1e-6f, theta = 1e7f;
    std::vector<int> target_layers;
    QMat fc, hproj;
    float *hidden_norm = nullptr, *out_norm = nullptr;
    uint16_t *pred = nullptr, *succ = nullptr;        // selector codebooks, FP16 rows of rank
    std::vector<DfLayer> layers;
    int64_t* ring_pos = nullptr;
    // buffers
    float *feat = nullptr, *h = nullptr, *dyn = nullptr, *a = nullptr, *hf = nullptr, *logits = nullptr, *gate = nullptr;
    float* cand_v = nullptr;
    int* cand_i = nullptr;
    int* tok = nullptr;
    float* conf = nullptr;
    void* topk_scratch = nullptr;
};

std::string upper_type(uint32_t t) { return strata::ggml_type_name(t); }

uint16_t f32_to_f16(float f) {   // round to nearest even (normal range; subnormals flush, overflow saturates)
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const int32_t e = (int32_t) ((x >> 23) & 0xff) - 127 + 15;
    const uint32_t m = x & 0x7fffffu;
    if (e <= 0) return (uint16_t) sign;
    if (e >= 31) return (uint16_t) (sign | 0x7bffu);
    uint32_t h = sign | ((uint32_t) e << 10) | (m >> 13);
    const uint32_t rest = m & 0x1fffu;
    if (rest > 0x1000u || (rest == 0x1000u && (h & 1u))) ++h;
    return (uint16_t) h;
}

// F16 rows (n_out of n_in) -> Q8_0 blocks (ggml's quantize_row_q8_0_ref), on the host, a few threads
std::vector<uint8_t> f16_to_q8_0(const uint16_t* src, int64_t n_in, int64_t n_out) {
    std::vector<uint8_t> out((size_t) (n_in / 32) * n_out * 34);
    const int64_t nb = n_in / 32 * n_out;
    const int nt = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
    std::vector<std::thread> th;
    for (int t = 0; t < nt; ++t)
        th.emplace_back([&, t] {
            for (int64_t b = t; b < nb; b += nt) {
                float v[32], amax = 0.0f;
                for (int i = 0; i < 32; ++i) { v[i] = strata::fp16_to_fp32(src[b * 32 + i]); amax = std::max(amax, std::fabs(v[i])); }
                const float d = amax / 127.0f, id = d ? 1.0f / d : 0.0f;
                uint8_t* o = out.data() + b * 34;
                const uint16_t dh = f32_to_f16(d);
                std::memcpy(o, &dh, 2);
                for (int i = 0; i < 32; ++i) o[2 + i] = (uint8_t) (int8_t) std::lround(v[i] * id);
            }
        });
    for (auto& x : th) x.join();
    return out;
}

// A tensor of one GGUF file, found by name across the shards.
struct Found {
    const strata::GgufFile* file = nullptr;
    const strata::TensorInfo* tensor = nullptr;
};

}  // namespace

struct DenseModel::Impl {
    std::vector<std::unique_ptr<strata::GgufFile>> files;
    std::map<std::string, Found> index;
    std::vector<Layer> layers;
    Layer mtp;                                      // blk.<n_layer>: the nextn block (attn == true when loaded)
    Dflash df;
    std::unique_ptr<strata::GgufFile> df_file;
    std::vector<uint8_t*> ckpt;                      // pinned host checkpoint slots (allocated on first use)
    std::vector<void*> allocations;                 // everything cudaMalloc'ed here, freed in the destructor
    std::vector<void*> host_allocs;                 // weights that did not fit in VRAM: mapped pinned host memory
    uint64_t budget = UINT64_MAX;                   // bytes of weights that may go to VRAM
    uint64_t dev_used = 0, host_bytes = 0;
    int host_tensors = 0;
    cudaStream_t stream = nullptr;
    int chunk = 512;                                // the largest T of a forward pass
    int kv_mode = 0;                                // the K / V cache: 0 FP16, 1 K8V4 (STRATA_DENSE_KV / --kv)

    QMat embd;                                      // token_embd, on the device (rows dequantized per token)
    float* output_norm = nullptr;
    QMat head;
    QMat head_draft;                                // the head's first draft_vocab rows (the drafts' vocabulary)
    void* q8_1 = nullptr;                           // the GEMVs' quantized activations (kMaxCols columns)
    uint16_t* xh = nullptr;                         // the GEMMs' FP16 activations (chunk rows of the widest input)
    strata::prefill::Gemm gemm;
    int cur_T = 0;                                  // the T the prepared activation (q8_1 / xh) is for

    // activations (device, f32, chunk rows each)
    float *x = nullptr, *xn = nullptr, *mix = nullptr, *hn = nullptr;
    float *qkv = nullptr, *conv = nullptr, *ab = nullptr;
    float *o = nullptr, *z = nullptr, *y = nullptr, *ab_tmp = nullptr;
    float *q_full = nullptr, *qcur = nullptr, *kcur = nullptr, *vcur = nullptr, *attn = nullptr;
    float *ffn_g = nullptr, *ffn_u = nullptr, *ffn_h = nullptr;
    float* logits = nullptr;                        // kMaxCols rows of n_vocab
    float* attn_scratch = nullptr;
    float* conv_dummy = nullptr;
    int32_t* tok_dev = nullptr;
    // MTP
    float *mtp_x = nullptr, *mtp_cat = nullptr, *mtp_h = nullptr, *mtp_logits = nullptr;

    // ---- STRATA_DENSE_PROF=1: GPU time a stage (events around each), printed every 64 forward passes
    struct Prof {
        bool on = false;
        std::vector<cudaEvent_t> ev;
        std::vector<int> ev_name;                  // the stage an event pair closes (-1: an opening event)
        std::vector<std::string> names;
        std::vector<double> ms;
        std::vector<long long> calls;
        int used = 0, passes = 0;
        cudaEvent_t next(cudaStream_t s) {
            if (used == (int) ev.size()) { cudaEvent_t e; cudaEventCreate(&e); ev.push_back(e); ev_name.push_back(-1); }
            cudaEventRecord(ev[used], s);
            return ev[used++];
        }
        int id(const char* n) {
            for (size_t i = 0; i < names.size(); ++i) if (names[i] == n) return (int) i;
            names.push_back(n); ms.push_back(0); calls.push_back(0);
            return (int) names.size() - 1;
        }
    } prof;
    void mark(cudaStream_t s) { if (prof.on) { prof.next(s); prof.ev_name[prof.used - 1] = -1; } }
    void stage(const char* n, cudaStream_t s) {    // closes the stage begun at the last mark/stage
        if (!prof.on) return;
        prof.next(s);
        prof.ev_name[prof.used - 1] = prof.id(n);
    }
    void prof_flush(int T) {
        if (!prof.on) return;
        cudaEventSynchronize(prof.ev[prof.used - 1]);
        for (int i = 1; i < prof.used; ++i) {
            if (prof.ev_name[i] < 0) continue;
            float m = 0;
            cudaEventElapsedTime(&m, prof.ev[i - 1], prof.ev[i]);
            prof.ms[prof.ev_name[i]] += m;
            prof.calls[prof.ev_name[i]] += 1;
        }
        prof.used = 0;
        if (++prof.passes % 64 == 0) {
            double tot = 0;
            for (double m : prof.ms) tot += m;
            std::fprintf(stderr, "dense prof (last 64 passes, T=%d): %.2f ms a pass\n", T, tot / 64);
            for (size_t i = 0; i < prof.names.size(); ++i)
                std::fprintf(stderr, "  %-14s %8.3f ms a pass %5.1f%%\n", prof.names[i].c_str(), prof.ms[i] / 64,
                             tot > 0 ? 100.0 * prof.ms[i] / tot : 0.0);
            std::fill(prof.ms.begin(), prof.ms.end(), 0.0);
        }
    }

    // ---- helpers
    template <class T> bool alloc(T** out, uint64_t bytes, std::string& err, const char* what) {
        void* p = nullptr;
        const cudaError_t s = cudaMalloc(&p, bytes ? bytes : 16);
        if (s != cudaSuccess) {
            err = std::string("dense model: cannot allocate ") + what + " (" + std::to_string(bytes >> 20) + " MiB): " +
                  cudaGetErrorString(s);
            return false;
        }
        allocations.push_back(p);
        *out = (T*) p;
        return true;
    }

    const Found* find(const std::string& name) const {
        auto it = index.find(name);
        return it == index.end() ? nullptr : &it->second;
    }
    const Found* find_any(std::initializer_list<const char*> names, const std::string& prefix) const {
        for (const char* n : names)
            if (const Found* f = find(prefix + n)) return f;
        return nullptr;
    }

    // A float vector or matrix (F32, F16 or BF16 in the file) uploaded as F32.
    bool upload_floats(const Found* f, const std::string& name, uint64_t expect, float** out, std::string& err) {
        if (!f) { err = "dense model: missing tensor " + name; return false; }
        uint64_t n = 1;
        for (uint64_t d : f->tensor->shape) n *= d;
        if (expect && n != expect) {
            err = "dense model: " + name + " has " + std::to_string(n) + " values, expected " + std::to_string(expect);
            return false;
        }
        const uint8_t* src = f->file->tensor_data(*f->tensor);
        std::vector<float> host(n);
        switch (f->tensor->type) {
        case kF32: strata::dequantize_f32(src, host.data(), (int) n); break;
        case kF16: strata::dequantize_f16(src, host.data(), (int) n); break;
        case kBF16: strata::dequantize_bf16(src, host.data(), (int) n); break;
        default:
            if (n > (uint64_t) INT_MAX || !dequant_row(f->tensor->type, src, (int) n, host.data())) {
                err = "dense model: " + name + " is " + upper_type(f->tensor->type) +
                      ", which cannot be read as floats (expected F32, F16, BF16 or a quantized type)";
                return false;
            }
        }
        if (!alloc(out, n * 4, err, name.c_str())) return false;
        const cudaError_t s = cudaMemcpy(*out, host.data(), n * 4, cudaMemcpyHostToDevice);
        if (s != cudaSuccess) { err = "dense model: upload " + name + ": " + cudaGetErrorString(s); return false; }
        return true;
    }

    // A quantized matrix, kept in its GGUF blocks.
    // q35_mmtp's packed layout for the GEMVs (v100-skinny / NInfer's prepacking; STRATA_DENSE_PACK=0: GGUF rows);
    // a prompt chunk's GEMM dequantizes straight from it (q35_mmt_dequant_packed).
    bool pack_on = true;
    bool pack(QMat& m, bool wide, std::string& err) {
        (void) wide;
        if (!pack_on || !q35_mmt_supported(m.type) || m.n_in % 256) return true;
        const uint64_t pb = q35_mmt_packed_bytes(m.type, m.n_in, m.n_out);
        void* p = nullptr;
        if (cudaMalloc(&p, pb) != cudaSuccess) { cudaGetLastError(); return true; }   // (no room: GGUF rows)
        q35_mmt_pack(m.type, m.dev, p, m.n_in, m.n_out, stream);
        if (cudaStreamSynchronize(stream) != cudaSuccess) { err = "dense model: packing failed"; return false; }
        const uint64_t ob = native_mmvq_weight_bytes(m.type, m.n_in, m.n_out);
        cudaFree(m.dev);
        allocations.erase(std::find(allocations.begin(), allocations.end(), m.dev));
        allocations.push_back(p);
        dev_used += pb - ob;
        m.dev = p;
        m.packed = true;
        return true;
    }

    bool upload_mat(const Found* f, const std::string& name, int n_in, int n_out, QMat& out, uint64_t& total,
                    std::string& err, bool packable = true, bool wide = true) {
        if (!f) { err = "dense model: missing tensor " + name; return false; }
        const strata::TensorInfo& t = *f->tensor;
        if (t.shape.size() != 2 || t.shape[0] != (uint64_t) n_in || t.shape[1] != (uint64_t) n_out) {
            err = "dense model: " + name + " has shape [" + std::to_string(t.shape.empty() ? 0 : t.shape[0]) + ", " +
                  std::to_string(t.shape.size() > 1 ? t.shape[1] : 0) + "], expected [" + std::to_string(n_in) + ", " +
                  std::to_string(n_out) + "]";
            return false;
        }
        if (!native_mmvq_supported((int) t.type) || !dequant_f16_supported((int) t.type)) {
            err = "dense model: " + name + " is " + upper_type(t.type) + ", which the native GEMVs do not take. "
                  "Supported: " + kSupported + "";
            return false;
        }
        int elems = 0, block_bytes = 0;
        if (!strata::block_geometry(t.type, elems, block_bytes) || (uint64_t) n_in % (uint64_t) elems) {
            err = "dense model: " + name + ": invalid block geometry"; return false;
        }
        const uint64_t bytes = native_mmvq_weight_bytes((int) t.type, n_in, n_out);
        const uint64_t in_file = (uint64_t) n_in * n_out / (uint64_t) elems * (uint64_t) block_bytes;
        const uint64_t payload = f->file->file_size() - f->file->data_start();
        if (bytes != in_file || t.offset > payload || bytes > payload - t.offset) {
            err = "dense model: " + name + ": payload size mismatch or truncated file"; return false;
        }
        void* dev = nullptr;
        cudaError_t s = cudaSuccess;
        if (dev_used + bytes <= budget) {
            s = cudaMalloc(&dev, bytes);
            if (s == cudaSuccess) s = cudaMemcpy(dev, f->file->tensor_data(t), bytes, cudaMemcpyHostToDevice);
            if (s != cudaSuccess) {                 // the card is fuller than cudaMemGetInfo said: spill this one
                if (dev) cudaFree(dev);
                dev = nullptr;
                cudaGetLastError();
            } else {
                allocations.push_back(dev);
                dev_used += bytes;
            }
        }
        if (!dev) {
            // Host-resident: pinned + mapped, so the GEMV kernel reads the blocks over PCIe on every token.
            void* host = nullptr;
            s = cudaHostAlloc(&host, bytes, cudaHostAllocMapped);
            if (s == cudaSuccess) s = cudaHostGetDevicePointer(&dev, host, 0);
            if (s != cudaSuccess) {
                if (host) cudaFreeHost(host);
                err = "dense model: " + name + " fits neither in VRAM nor in pinned host memory (" +
                      std::to_string(bytes >> 20) + " MiB): " + cudaGetErrorString(s);
                return false;
            }
            std::memcpy(host, f->file->tensor_data(t), bytes);
            host_allocs.push_back(host);
            host_bytes += bytes;
            ++host_tensors;
        }
        const bool in_vram = std::find(allocations.begin(), allocations.end(), dev) != allocations.end();
        out = QMat{dev, (int) t.type, n_in, n_out};
        total += bytes;
        return !(packable && in_vram) || pack(out, wide, err);
    }

    // ---- the projections of T token rows (x dense rows of n_in): prep() once per input, then mm() per matrix
    // up to 8 rows: the tensor-core GEMV (q35_mmt) for the formats it takes - from mmt_min rows, or from 1 with
    // STRATA_DENSE_MMT1 - else native_mmvq's dp4a GEMV (its q8_1 activations made on first use)
    bool mmt_on = true, mmt1 = false;
    int mmt_min = 4;                                // (dp4a is faster up to 3 rows on a V100)
    uint16_t* xh8 = nullptr;
    float* mmt_part = nullptr;
    const float* cur_x = nullptr;
    int cur_n = 0;
    bool q8_ready = false, h8_ready = false;
    void prep(const float* x, int n_in, int T) {
        cur_T = T; cur_x = x; cur_n = n_in; q8_ready = h8_ready = false;
        if (T > kMaxCols) q35_to_half(x, xh, (int64_t) T * n_in, stream);
    }
    void mm(const QMat& w, float* y) {
        if (cur_T > kMaxCols) {
            if (w.packed) {   // dequantized to FP16 straight from the packed layout, row slices that fit the scratch
                const int64_t step = std::max<int64_t>(32, gemm.scratch_elems() / w.n_in / 32 * 32);
                for (int64_t r0 = 0; r0 < w.n_out; r0 += step) {
                    const int rows = (int) std::min<int64_t>(step, w.n_out - r0);
                    q35_mmt_dequant_packed(w.type, w.dev, w.n_in, w.n_out, (int) r0, rows, gemm.scratch(), stream);
                    gemm.f16(xh, gemm.scratch(), y + r0, cur_T, rows, w.n_in, w.n_out);
                }
            } else {
                gemm.native(xh, w.type, w.dev, y, cur_T, w.n_out, w.n_in);
            }
            return;
        }
        if (w.packed) {
            if (!h8_ready) { q35_mmt_prep(cur_x, cur_n, cur_T, xh8, stream); h8_ready = true; }
            q35_mmtp(w.type, w.dev, xh8, y, w.n_in, w.n_out, cur_T, mmt_part, stream);
            return;
        }
        if (mmt_on && (cur_T >= mmt_min || mmt1) && q35_mmt_supported(w.type)) {
            if (!h8_ready) { q35_mmt_prep(cur_x, cur_n, cur_T, xh8, stream); h8_ready = true; }
            q35_mmt(w.type, w.dev, xh8, y, w.n_in, w.n_out, cur_T, mmt_part, stream);
            return;
        }
        if (!q8_ready) { native_quantize_q8_1(cur_x, q8_1, cur_n, cur_T, stream); q8_ready = true; }
        native_mmvq(w.type, w.dev, q8_1, y, w.n_in, w.n_out, cur_T, stream);
    }
};

namespace {

// Host-side dequantization of one row (n values, whole blocks).
bool dequant_row(uint32_t type, const uint8_t* src, int n, float* out) {
    switch (type) {
    case kF32: strata::dequantize_f32(src, out, n); return true;
    case kF16: strata::dequantize_f16(src, out, n); return true;
    case kBF16: strata::dequantize_bf16(src, out, n); return true;
    default: break;
    }
    int elems = 0, bytes = 0;
    if (!strata::block_geometry(type, elems, bytes) || n % elems) return false;
    for (int b = 0; b < n / elems; ++b) {
        const uint8_t* blk = src + (size_t) b * bytes;
        float* o = out + (size_t) b * elems;
        switch (type) {
        case 2: strata::dequantize_q4_0(blk, o); break;
        case 6: strata::dequantize_q5_0(blk, o); break;
        case 8: strata::dequantize_q8_0(blk, o); break;
        case 11: strata::dequantize_q3_K(blk, o); break;
        case 12: strata::dequantize_q4_K(blk, o); break;
        case 13: strata::dequantize_q5_K(blk, o); break;
        case 14: strata::dequantize_q6_K(blk, o); break;
        case 20: strata::dequantize_iq4_nl(blk, o); break;
        case 23: strata::dequantize_iq4_xs(blk, o); break;
        case 42: strata::dequantize_q2_0(blk, o); break;
        default: return false;
        }
    }
    return true;
}

// llama.cpp names a split file <base>-00001-of-0000N.gguf
std::vector<std::string> shard_paths(const std::string& path) {
    static const std::regex re(R"((.*-)(\d{5})-of-(\d{5})\.gguf$)");
    std::smatch m;
    if (!std::regex_match(path, m, re)) return {path};
    const int n = std::stoi(m[3].str());
    std::vector<std::string> out;
    for (int i = 1; i <= n; ++i) {
        char buf[16];
        std::snprintf(buf, sizeof buf, "%05d", i);
        out.push_back(m[1].str() + buf + "-of-" + m[3].str() + ".gguf");
    }
    return out;
}

// integer metadata that may be a scalar or (per layer) an array: the first element
bool meta_int(const strata::GgufFile& g, const std::string& key, int64_t& out) {
    const strata::MetaValue* v = g.get(key);
    if (!v) return false;
    if (v->type == strata::MetaType::ARRAY) {
        if (v->items.empty()) return false;
        out = (int64_t) v->items[0].u;
        return true;
    }
    out = (int64_t) v->u;
    return true;
}

}  // namespace

DenseModel::DenseModel() : impl_(new Impl) {}

DenseModel::~DenseModel() {
    if (!impl_) return;
    if (impl_->stream) cudaStreamSynchronize(impl_->stream);
    for (void* p : impl_->allocations) cudaFree(p);
    for (void* p : impl_->host_allocs) cudaFreeHost(p);
    for (uint8_t* p : impl_->ckpt) if (p) cudaFreeHost(p);
    if (impl_->stream) cudaStreamDestroy(impl_->stream);
}

float* DenseModel::logits() const { return impl_->logits; }
void* DenseModel::stream() const { return impl_->stream; }
const float* DenseModel::hidden_norm() const { return impl_->hn; }
int DenseModel::chunk() const { return impl_->chunk; }
int DenseModel::draft_vocab() const { return impl_->head_draft.n_out; }
bool DenseModel::has_mtp() const { return impl_->mtp.attn; }
float* DenseModel::mtp_logits() const { return impl_->mtp_logits; }
const float* DenseModel::mtp_hidden() const { return impl_->mtp_h; }

bool DenseModel::load(const std::string& path, int64_t max_context, std::string& err, int chunk) {
    Impl& I = *impl_;
    I.chunk = std::max(kMaxCols + 1, chunk);
    if (const char* e = std::getenv("STRATA_DENSE_PACK")) I.pack_on = std::atoi(e) != 0;
    if (const char* e = std::getenv("STRATA_DENSE_KV")) {
        if (std::string(e) == "k8v4") I.kv_mode = 1;
        else if (std::string(e) != "fp16") { err = std::string("dense model: STRATA_DENSE_KV=") + e + " (fp16 | k8v4)"; return false; }
    }
    I.prof.on = std::getenv("STRATA_DENSE_PROF") != nullptr && std::atoi(std::getenv("STRATA_DENSE_PROF")) != 0;
    try {
        // ---- 1. the files and the tensor index
        for (const std::string& p : shard_paths(path)) I.files.emplace_back(new strata::GgufFile(p));
        for (const auto& f : I.files)
            for (const auto& t : f->tensors()) I.index[t.name] = Found{f.get(), &t};
        const strata::GgufFile& g0 = *I.files.front();

        // ---- 2. the configuration, from the model's own metadata
        const strata::MetaValue* arch = g0.get("general.architecture");
        if (!arch) { err = "dense model: missing general.architecture"; return false; }
        cfg_.arch = arch->s;
        const std::string a = cfg_.arch + ".";
        if (cfg_.arch.rfind("qwen3", 0) != 0) {
            err = "dense model: architecture '" + cfg_.arch + "' is not a qwen35 dense model";
            return false;
        }
        if (g0.get(a + "expert_count") && g0.get(a + "expert_count")->u > 0) {
            err = "dense model: this GGUF is a MoE model; run it with `strata`, not `strata-dense`";
            return false;
        }
        int64_t v = 0;
        auto need = [&](const char* key, int& dst) {
            if (!meta_int(g0, a + key, v)) { err = std::string("dense model: missing ") + a + key; return false; }
            dst = (int) v;
            return true;
        };
        if (!need("block_count", cfg_.n_layer) || !need("embedding_length", cfg_.n_embd) ||
            !need("feed_forward_length", cfg_.n_ff) || !need("attention.head_count", cfg_.n_head) ||
            !need("attention.head_count_kv", cfg_.n_head_kv) || !need("attention.key_length", cfg_.head_dim) ||
            !need("rope.dimension_count", cfg_.n_rot) || !need("ssm.conv_kernel", cfg_.ssm_d_conv) ||
            !need("ssm.state_size", cfg_.ssm_state) || !need("ssm.group_count", cfg_.ssm_k_heads) ||
            !need("ssm.time_step_rank", cfg_.ssm_v_heads))
            return false;
        // block_count includes the MTP/nextn layer(s) stored at the end of the file: they are not part of the main
        // forward pass (llama.cpp uses n_layer_all - n_layer_nextn), running them corrupts the hidden state.
        if (meta_int(g0, a + "nextn_predict_layers", v) && v > 0 && v < cfg_.n_layer) {
            cfg_.n_nextn = (int) v;
            cfg_.n_layer -= (int) v;
        }
        if (const auto* e = g0.get(a + "attention.layer_norm_rms_epsilon")) cfg_.rms_eps = (float) e->num();
        if (const auto* e = g0.get(a + "rope.freq_base")) cfg_.rope_base = (float) e->num();
        if (const auto* e = g0.get("tokenizer.ggml.eos_token_id")) cfg_.eos_id = (int32_t) e->u;
        if (meta_int(g0, a + "ssm.inner_size", v) && v != cfg_.value_dim()) {
            err = "dense model: ssm.inner_size " + std::to_string(v) + " != state_size * time_step_rank";
            return false;
        }
        if (cfg_.ssm_state != 128 || cfg_.ssm_d_conv != 4 || cfg_.ssm_v_heads % cfg_.ssm_k_heads != 0) {
            err = "dense model: the GDN kernels are written for state_size 128 and conv_kernel 4";
            return false;
        }
        if (cfg_.head_dim != 256 || cfg_.n_head % cfg_.n_head_kv != 0 || cfg_.n_head / cfg_.n_head_kv > 8 ||
            cfg_.n_rot > cfg_.head_dim || cfg_.n_rot % 2 != 0) {
            err = "dense model: unsupported attention geometry (head size 256, up to 8 query heads a KV head)";
            return false;
        }
        if (cfg_.n_embd % 256 != 0 || cfg_.n_ff % 256 != 0 || cfg_.value_dim() % 256 != 0 ||
            (cfg_.n_head * cfg_.head_dim) % 256 != 0) {
            err = "dense model: every GEMV input width must be a multiple of 256 (K-quant super-block)";
            return false;
        }
        const Found* tok = I.find("token_embd.weight");
        if (!tok || tok->tensor->shape.size() != 2 || tok->tensor->shape[0] != (uint64_t) cfg_.n_embd) {
            err = "dense model: token_embd.weight missing or not [n_embd, n_vocab]";
            return false;
        }
        cfg_.n_vocab = (int) tok->tensor->shape[1];
        max_context_ = max_context;
        if (max_context < 16) { err = "dense model: context must be at least 16"; return false; }

        cudaError_t cs = cudaStreamCreate(&I.stream);
        if (cs != cudaSuccess) { err = std::string("dense model: CUDA setup: ") + cudaGetErrorString(cs); return false; }

        // ---- 3. the layers
        uint64_t total = 0;
        const int C = cfg_.conv_channels(), V = cfg_.value_dim(), H = cfg_.n_head, HK = cfg_.n_head_kv, D = cfg_.head_dim;
        const int E = cfg_.n_embd, NL = cfg_.n_layer;
        const bool want_mtp = cfg_.n_nextn > 0 && I.find("blk." + std::to_string(NL) + ".nextn.eh_proj.weight") &&
                              std::getenv("STRATA_DENSE_NO_MTP") == nullptr;
        {   // How much of the weights VRAM can take: what is free, minus the KV caches, the activations and a margin.
            int n_attn = want_mtp ? 1 : 0;
            for (int l = 0; l < NL; ++l)
                if (I.find("blk." + std::to_string(l) + ".attn_q.weight")) ++n_attn;
            const uint64_t kv_need = (uint64_t) n_attn * q35_kv_bytes(I.kv_mode, HK, D, (int) max_context);
            const uint64_t act = (uint64_t) I.chunk * (uint64_t) (8 * E + 3 * C + 3 * V + 4 * H * D + 3 * cfg_.n_ff) * 4 +
                                 (uint64_t) cfg_.n_ff * E * 2 + (uint64_t) kMaxCols * cfg_.n_vocab * 4 * 2 +
                                 (uint64_t) q35_attn_scratch_floats(kMaxCols, H, D, (int) max_context) * 4;
            size_t free_b = 0, total_b = 0;
            cudaMemGetInfo(&free_b, &total_b);
            const uint64_t margin = 768ull << 20;
            if (const char* e = std::getenv("STRATA_DENSE_GPU_MIB")) {       // explicit cap for the weights, in MiB
                I.budget = (uint64_t) std::atoll(e) << 20;
            } else if ((uint64_t) free_b > kv_need + act + margin) {
                I.budget = (uint64_t) free_b - kv_need - act - margin;
            } else {
                err = "dense model: the KV cache for a context of " + std::to_string(max_context) + " needs " +
                      std::to_string(kv_need >> 20) + " MiB, but only " + std::to_string(free_b >> 20) +
                      " MiB of VRAM are free. Use a shorter --context.";
                return false;
            }
            std::fprintf(stderr, "strata-dense: VRAM %zu MiB free of %zu; KV cache %llu MiB (context %lld), activations "
                         "%llu MiB; %llu MiB of weights may go to the GPU\n",
                         free_b >> 20, total_b >> 20, (unsigned long long) (kv_need >> 20), (long long) max_context,
                         (unsigned long long) (act >> 20), (unsigned long long) (I.budget >> 20));
        }
        auto load_attn = [&](Layer& L, const std::string& p) {
            if (!I.upload_mat(I.find(p + "attn_q.weight"), p + "attn_q.weight", E, H * 2 * D, L.q, total, err) ||
                !I.upload_mat(I.find(p + "attn_k.weight"), p + "attn_k.weight", E, HK * D, L.k, total, err) ||
                !I.upload_mat(I.find(p + "attn_v.weight"), p + "attn_v.weight", E, HK * D, L.v, total, err) ||
                !I.upload_mat(I.find(p + "attn_output.weight"), p + "attn_output.weight", H * D, E, L.o, total, err))
                return false;
            if (!I.upload_floats(I.find(p + "attn_q_norm.weight"), p + "attn_q_norm.weight", D, &L.q_norm, err) ||
                !I.upload_floats(I.find(p + "attn_k_norm.weight"), p + "attn_k_norm.weight", D, &L.k_norm, err))
                return false;
            L.kv.mode = I.kv_mode;
            L.kv.max_ctx = (int) max_context;
            const uint64_t cells = (uint64_t) HK * (uint64_t) max_context;
            if (I.kv_mode == 0) {
                if (!I.alloc(&L.kv.k, cells * D * 2, err, "K cache") || !I.alloc(&L.kv.v, cells * D * 2, err, "V cache"))
                    return false;
            } else if (!I.alloc(&L.kv.k, cells * D, err, "K cache") || !I.alloc(&L.kv.v, cells * D / 2, err, "V cache") ||
                       !I.alloc(&L.kv.ks, cells * (D / 32) * 2, err, "K scales") ||
                       !I.alloc(&L.kv.vs, cells * (D / 32) * 2, err, "V scales")) {
                return false;
            }
            kv_bytes_ += q35_kv_bytes(I.kv_mode, HK, D, (int) max_context);
            return true;
        };
        auto load_common = [&](Layer& L, const std::string& p) {
            if (!I.upload_floats(I.find(p + "attn_norm.weight"), p + "attn_norm.weight", E, &L.attn_norm, err)) return false;
            if (!I.upload_floats(I.find_any({"post_attention_norm.weight", "ffn_norm.weight"}, p), p + "post_attention_norm.weight",
                                 E, &L.post_norm, err)) return false;
            return I.upload_mat(I.find(p + "ffn_gate.weight"), p + "ffn_gate.weight", E, cfg_.n_ff, L.ffn_gate, total, err) &&
                   I.upload_mat(I.find(p + "ffn_up.weight"), p + "ffn_up.weight", E, cfg_.n_ff, L.ffn_up, total, err) &&
                   I.upload_mat(I.find(p + "ffn_down.weight"), p + "ffn_down.weight", cfg_.n_ff, E, L.ffn_down, total, err);
        };
        I.layers.assign((size_t) NL, Layer{});
        for (int l = 0; l < NL; ++l) {
            Layer& L = I.layers[(size_t) l];
            const std::string p = "blk." + std::to_string(l) + ".";
            const std::string at = "layer " + std::to_string(l) + ": ";
            const bool has_gdn = I.find(p + "attn_qkv.weight") != nullptr;
            const bool has_attn = I.find(p + "attn_q.weight") != nullptr;
            if (has_gdn == has_attn) { err = "dense model: " + at + "needs exactly one of attn_qkv.weight / attn_q.weight"; return false; }
            L.attn = has_attn;
            if (!load_common(L, p)) return false;
            if (!L.attn) {
                if (!I.upload_mat(I.find(p + "attn_qkv.weight"), p + "attn_qkv.weight", E, C, L.qkv, total, err) ||
                    !I.upload_mat(I.find(p + "attn_gate.weight"), p + "attn_gate.weight", E, V, L.z, total, err) ||
                    !I.upload_mat(I.find(p + "ssm_out.weight"), p + "ssm_out.weight", V, E, L.ssm_out, total, err))
                    return false;
                if (!I.upload_floats(I.find(p + "ssm_conv1d.weight"), p + "ssm_conv1d.weight", (uint64_t) C * cfg_.ssm_d_conv, &L.conv_w, err) ||
                    !I.upload_floats(I.find_any({"ssm_dt.bias", "ssm_dt"}, p), p + "ssm_dt.bias", cfg_.ssm_v_heads, &L.dt, err) ||
                    !I.upload_floats(I.find_any({"ssm_a", "ssm_a.weight"}, p), p + "ssm_a", cfg_.ssm_v_heads, &L.ssm_a, err) ||
                    !I.upload_floats(I.find(p + "ssm_norm.weight"), p + "ssm_norm.weight", cfg_.ssm_state, &L.ssm_norm, err))
                    return false;
                {   // alpha / beta: a float matrix in some files, Q8_0 (or another block type) in others
                    const Found* fa = I.find(p + "ssm_alpha.weight");
                    const Found* fb = I.find(p + "ssm_beta.weight");
                    if (!fa || !fb) { err = "dense model: " + at + "missing ssm_alpha.weight / ssm_beta.weight"; return false; }
                    auto is_float = [](uint32_t t) { return t == kF32 || t == kF16 || t == kBF16; };
                    if (is_float(fa->tensor->type) != is_float(fb->tensor->type)) {
                        err = "dense model: " + at + "ssm_alpha and ssm_beta have different kinds of type"; return false;
                    }
                    L.ab_quant = !is_float(fa->tensor->type);
                    if (L.ab_quant) {
                        if (!I.upload_mat(fa, p + "ssm_alpha.weight", E, cfg_.ssm_v_heads, L.alpha_q, total, err) ||
                            !I.upload_mat(fb, p + "ssm_beta.weight", E, cfg_.ssm_v_heads, L.beta_q, total, err))
                            return false;
                    } else {   // one matrix for both: [alpha rows | beta rows]
                        float *wa = nullptr, *wb = nullptr;
                        const uint64_t n1 = (uint64_t) E * cfg_.ssm_v_heads;
                        if (!I.upload_floats(fa, p + "ssm_alpha.weight", n1, &wa, err) ||
                            !I.upload_floats(fb, p + "ssm_beta.weight", n1, &wb, err) ||
                            !I.alloc(&L.ab_w, 2 * n1 * 4, err, "alpha/beta"))
                            return false;
                        cudaMemcpy(L.ab_w, wa, n1 * 4, cudaMemcpyDeviceToDevice);
                        cudaMemcpy(L.ab_w + n1, wb, n1 * 4, cudaMemcpyDeviceToDevice);
                        for (void* t : {(void*) wa, (void*) wb}) {
                            cudaFree(t);
                            I.allocations.erase(std::find(I.allocations.begin(), I.allocations.end(), t));
                        }
                    }
                }
                const uint64_t sbytes = (uint64_t) cfg_.ssm_state * cfg_.ssm_v_heads * cfg_.ssm_state * 4;
                const uint64_t cbytes = (uint64_t) C * (cfg_.ssm_d_conv - 1) * 4;
                if (!I.alloc(&L.state, sbytes, err, "GDN state") || !I.alloc(&L.conv_state, cbytes, err, "conv state") ||
                    !I.alloc(&L.st_raw, (uint64_t) kMaxCols * C * 4, err, "verify stash") ||
                    !I.alloc(&L.st_conv, (uint64_t) kMaxCols * C * 4, err, "verify stash") ||
                    !I.alloc(&L.st_ab, (uint64_t) kMaxCols * 2 * cfg_.ssm_v_heads * 4, err, "verify stash"))
                    return false;
            } else if (!load_attn(L, p)) {
                return false;
            }
        }
        if (want_mtp) {   // the nextn block: a full-attention layer behind eh_proj . [enorm(emb) | hnorm(h)]
            Layer& M = I.mtp;
            const std::string p = "blk." + std::to_string(NL) + ".";
            if (!load_common(M, p) || !load_attn(M, p) ||
                !I.upload_mat(I.find(p + "nextn.eh_proj.weight"), p + "nextn.eh_proj.weight", 2 * E, E, M.eh_proj, total, err) ||
                !I.upload_floats(I.find(p + "nextn.enorm.weight"), p + "nextn.enorm.weight", E, &M.enorm, err) ||
                !I.upload_floats(I.find(p + "nextn.hnorm.weight"), p + "nextn.hnorm.weight", E, &M.hnorm, err))
                return false;
            const Found* hn = I.find(p + "nextn.shared_head_norm.weight");
            if (!I.upload_floats(hn ? hn : I.find("output_norm.weight"), "mtp head norm", E, &M.head_norm, err)) return false;
            M.attn = true;
        }

        // ---- 4. the input and output side
        if (!I.upload_mat(tok, "token_embd.weight", E, cfg_.n_vocab, I.embd, total, err, false)) return false;
        if (!I.upload_floats(I.find("output_norm.weight"), "output_norm.weight", E, &I.output_norm, err)) return false;
        const Found* head = I.find("output.weight");
        if (!head) head = tok;                                   // tied embeddings
        if (!I.upload_mat(head, head == tok ? "token_embd.weight (tied head)" : "output.weight", E, cfg_.n_vocab,
                          I.head, total, err, head != tok, false))
            return false;
        I.head_draft = I.head;
        if (const char* e = std::getenv("STRATA_DENSE_DRAFT_VOCAB")) {
            const int dv = std::atoi(e);
            if (dv > 0 && dv < cfg_.n_vocab) I.head_draft.n_out = dv / 128 * 128;   // (rows are contiguous blocks)
        }
        weight_bytes_ = total;
        host_bytes_ = I.host_bytes;
        if (I.host_tensors)
            std::fprintf(stderr, "strata-dense: %d tensors (%.2f GiB of %.2f) live in host memory and are read over PCIe "
                         "on every token - a shorter --context or a smaller quant puts more on the GPU\n",
                         I.host_tensors, I.host_bytes / 1073741824.0, total / 1073741824.0);

        // ---- 5. activations and scratch
        const uint64_t R = (uint64_t) I.chunk, f = 4;
        const int max_in = std::max({E, cfg_.n_ff, V, H * D, 2 * E, 5 * E});   // (5 E: the DFlash2 drafter's fc)
        bool ok = I.alloc(&I.q8_1, native_q8_1_bytes(max_in, kMaxCols), err, "q8_1 scratch") &&
                  I.alloc(&I.xh, R * max_in * 2, err, "fp16 activations") &&
                  I.alloc(&I.x, R * E * f, err, "x") && I.alloc(&I.xn, R * E * f, err, "xn") &&
                  I.alloc(&I.mix, R * E * f, err, "mix") && I.alloc(&I.hn, R * E * f, err, "hidden") &&
                  I.alloc(&I.qkv, R * C * f, err, "qkv") && I.alloc(&I.conv, R * C * f, err, "conv") &&
                  I.alloc(&I.ab, R * 2 * cfg_.ssm_v_heads * f, err, "alpha/beta") && I.alloc(&I.o, R * V * f, err, "o") &&
                  I.alloc(&I.z, R * V * f, err, "z") && I.alloc(&I.y, R * V * f, err, "y") &&
                  I.alloc(&I.q_full, R * H * 2 * D * f, err, "q_full") && I.alloc(&I.qcur, R * H * D * f, err, "qcur") &&
                  I.alloc(&I.kcur, R * HK * D * f, err, "kcur") && I.alloc(&I.vcur, R * HK * D * f, err, "vcur") &&
                  I.alloc(&I.attn, R * H * D * f, err, "attn") &&
                  I.alloc(&I.ffn_g, R * cfg_.n_ff * f, err, "ffn_g") && I.alloc(&I.ffn_u, R * cfg_.n_ff * f, err, "ffn_u") &&
                  I.alloc(&I.ffn_h, R * cfg_.n_ff * f, err, "ffn_h") &&
                  I.alloc(&I.logits, (uint64_t) kMaxCols * cfg_.n_vocab * f, err, "logits") &&
                  I.alloc(&I.attn_scratch, (uint64_t) q35_attn_scratch_floats(kMaxCols, H, D, (int) max_context) * f, err,
                          "attention scratch") &&
                  I.alloc(&I.conv_dummy, (uint64_t) kMaxCols * C * f, err, "conv scratch") &&
                  I.alloc(&I.ab_tmp, R * 2 * cfg_.ssm_v_heads * f, err, "alpha/beta") &&
                  I.alloc(&I.tok_dev, R * 4, err, "tokens") &&
                  I.alloc(&I.mtp_x, (uint64_t) kMaxCols * E * f, err, "mtp") &&
                  I.alloc(&I.mtp_cat, (uint64_t) R * 2 * E * f, err, "mtp") &&
                  I.alloc(&I.mtp_h, R * E * f, err, "mtp") &&
                  I.alloc(&I.mtp_logits, (uint64_t) kMaxCols * cfg_.n_vocab * f, err, "mtp logits") &&
                  I.alloc(&I.xh8, (uint64_t) 8 * max_in * 2, err, "fp16 rows") &&
                  I.alloc(&I.mmt_part, (uint64_t) q35_mmt_part_floats(std::max({cfg_.n_vocab, cfg_.n_ff, H * 2 * D})) * f,
                          err, "GEMV partials");
        if (!ok) return false;
        if (!I.gemm.init(I.stream, (int64_t) cfg_.n_ff * E, err)) return false;
        if (const char* e = std::getenv("STRATA_DENSE_MULTI_EXACT")) native_mmvq_set_multi_exact(std::atoi(e) != 0);
        if (const char* e = std::getenv("STRATA_DENSE_MMT")) I.mmt_on = std::atoi(e) != 0;
        if (const char* e = std::getenv("STRATA_DENSE_MMT1")) I.mmt1 = std::atoi(e) != 0;
        if (const char* e = std::getenv("STRATA_DENSE_MMT_MIN")) I.mmt_min = std::max(1, std::atoi(e));

        reset();
        std::string serr;
        if (!sync(serr)) { err = serr; return false; }
        std::fprintf(stderr, "strata-dense: %d layers%s, prompt chunks of %d tokens\n", NL,
                     I.mtp.attn ? " + the MTP block" : "", I.chunk);
        return true;
    } catch (const std::exception& e) {
        err = std::string("dense model: ") + e.what();
        return false;
    }
}

void DenseModel::reset() {
    Impl& I = *impl_;
    const uint64_t sbytes = (uint64_t) cfg_.ssm_state * cfg_.ssm_v_heads * cfg_.ssm_state * 4;
    const uint64_t cbytes = (uint64_t) cfg_.conv_channels() * (cfg_.ssm_d_conv - 1) * 4;
    for (Layer& L : I.layers) {
        if (L.state) cudaMemsetAsync(L.state, 0, sbytes, I.stream);
        if (L.conv_state) cudaMemsetAsync(L.conv_state, 0, cbytes, I.stream);
    }
    if (I.df.on) cudaMemsetAsync(I.df.ring_pos, 0xff, (size_t) I.df.window * 8, I.stream);   // the drafter's context
    pos_ = 0;
    pending_ = 0;
}

bool DenseModel::sync(std::string& err) {
    const cudaError_t s = cudaStreamSynchronize(impl_->stream);
    if (s != cudaSuccess) { err = std::string("dense model: ") + cudaGetErrorString(s); return false; }
    return true;
}

namespace {

// The attention layer's mixer for T rows at p0 (xn prepared): mix = attn_output(gated attention)
void attn_mixer(DenseModel::Impl& I, const DenseConfig& c, Layer& L, int64_t p0, int T, int max_ctx) {
    const int H = c.n_head, HK = c.n_head_kv, D = c.head_dim;
    I.mm(L.q, I.q_full);
    I.mm(L.k, I.kcur);
    I.mm(L.v, I.vcur);
    I.stage("attn_qkv", I.stream);
    q35_attn_prep(I.q_full, H * 2 * D, I.kcur, I.vcur, HK * D, L.q_norm, L.k_norm, c.rms_eps, c.rope_base, c.n_rot,
                  (int) p0, T, H, HK, D, I.qcur, L.kv, I.stream);
    (void) max_ctx;
    q35_attn(I.qcur, L.kv, (int) p0, T, H, HK, D, I.q_full, H * 2 * D, I.attn, I.attn_scratch,
             T <= kMaxCols ? 0 : 1, I.stream);
    I.stage("attn_core", I.stream);
    I.prep(I.attn, H * D, T);
    I.mm(L.o, I.mix);
    I.stage("attn_out", I.stream);
}

// x += ffn(rms(x, post_norm)), T rows
void ffn(DenseModel::Impl& I, const DenseConfig& c, Layer& L, float* x, int T) {
    const int E = c.n_embd;
    q35_rms_rows(x, E, L.post_norm, I.xn, nullptr, T, E, c.rms_eps, I.stream);
    I.prep(I.xn, E, T);
    I.stage("norm_prep", I.stream);
    I.mm(L.ffn_gate, I.ffn_g);
    I.mm(L.ffn_up, I.ffn_u);
    I.stage("ffn_gate_up", I.stream);
    dense_swiglu(I.ffn_g, I.ffn_u, I.ffn_h, (int64_t) T * c.n_ff, I.stream);
    I.prep(I.ffn_h, c.n_ff, T);
    I.mm(L.ffn_down, I.mix);
    q35_add(x, I.mix, (int64_t) T * E, I.stream);
}

}  // namespace

bool DenseModel::forward(const int32_t* tokens, int T, bool verify, int want, std::string& err) {
    Impl& I = *impl_;
    if (T < 1 || T > I.chunk) { err = "dense model: a forward pass takes 1.." + std::to_string(I.chunk) + " tokens"; return false; }
    if (verify && T > kMaxCols) { err = "dense model: a verify pass takes up to 8 tokens"; return false; }
    if (want == kAll && T > kMaxCols) { err = "dense model: logits of every row up to 8 rows"; return false; }
    if (pending_) { err = "dense model: a verify pass waits for commit()"; return false; }
    if (pos_ + T > max_context_) { err = "dense model: the context is full"; return false; }
    for (int t = 0; t < T; ++t)
        if (tokens[t] < 0 || tokens[t] >= cfg_.n_vocab) { err = "dense model: token id outside the vocabulary"; return false; }
    cudaStream_t s = I.stream;
    const int C = cfg_.conv_channels(), V = cfg_.value_dim(), E = cfg_.n_embd;
    const int KH = cfg_.ssm_k_heads, VH = cfg_.ssm_v_heads;
    const float eps = cfg_.rms_eps;
    const int64_t p0 = pos_;
    try {
        // ---- the embedding rows, dequantized on the device
        I.mark(s);
        for (int t = 0; t < T; ++t)
            dequant_f32(I.embd.type, I.embd.dev, tokens[t], 1, E, I.x + (size_t) t * E, s);
        I.stage("embed", s);

        for (int l = 0; l < cfg_.n_layer; ++l) {
            Layer& L = I.layers[(size_t) l];
            q35_rms_rows(I.x, E, L.attn_norm, I.xn, nullptr, T, E, eps, s);
            I.prep(I.xn, E, T);
            I.stage("norm_prep", s);
            if (!L.attn) {
                // ---- gated delta net: a verify pass keeps the recurrence's inputs for commit()
                float* raw = verify ? L.st_raw : I.qkv;
                float* cv = verify ? L.st_conv : I.conv;
                float* ab = verify ? L.st_ab : I.ab;    // rows [alpha | beta] -> [g | b]
                I.mm(L.qkv, raw);
                I.mm(L.z, I.z);
                I.stage("gdn_qkvz", s);
                if (L.ab_quant) {   // (the quantized pair: two GEMVs into the halves of a row)
                    I.mm(L.alpha_q, I.ab_tmp);
                    I.mm(L.beta_q, I.ab_tmp + (size_t) T * VH);
                    cudaMemcpy2DAsync(ab, (size_t) 2 * VH * 4, I.ab_tmp, (size_t) VH * 4, (size_t) VH * 4, T,
                                      cudaMemcpyDeviceToDevice, s);
                    cudaMemcpy2DAsync(ab + VH, (size_t) 2 * VH * 4, I.ab_tmp + (size_t) T * VH, (size_t) VH * 4,
                                      (size_t) VH * 4, T, cudaMemcpyDeviceToDevice, s);
                } else {
                    q35_gemv_f32_cols(L.ab_w, I.xn, E, ab, E, 2 * VH, T, s);
                }
                I.stage("gdn_ab", s);
                q35_conv_seq(L.conv_state, raw, C, L.conv_w, cv, T, C, !verify, s);
                q35_qk_l2(cv, C, T, 2 * KH, eps, s);
                q35_gate_beta(ab, ab + VH, 2 * VH, L.dt, L.ssm_a, T, VH, s);
                I.stage("gdn_conv", s);
                q35_gdn_seq(L.state, cv, C, ab, ab + VH, 2 * VH, I.o, T, KH, VH, !verify, s);
                I.stage("gdn_rec", s);
                q35_gdn_out(I.o, I.z, L.ssm_norm, I.y, T * VH, eps, s);
                I.prep(I.y, V, T);
                I.mm(L.ssm_out, I.mix);
                I.stage("gdn_out", s);
            } else {
                attn_mixer(I, cfg_, L, p0, T, (int) max_context_);
            }
            q35_add(I.x, I.mix, (int64_t) T * E, s);
            ffn(I, cfg_, L, I.x, T);
            I.stage("ffn", s);
            if (I.df.on)   // the drafter's features: the residual after its target layers, in their order
                for (size_t j = 0; j < I.df.target_layers.size(); ++j)
                    if (I.df.target_layers[j] == l)
                        cudaMemcpy2DAsync(I.df.feat + j * E, I.df.target_layers.size() * E * 4, I.x, (size_t) E * 4,
                                          (size_t) E * 4, T, cudaMemcpyDeviceToDevice, s);
        }
        // the final norm of every row (the MTP block reads them), then the logits asked for
        q35_rms_rows(I.x, E, I.output_norm, I.hn, nullptr, T, E, eps, s);
        if (want != kNone) {
            const int rows = want == kAll ? T : 1;
            I.prep(want == kAll ? I.hn : I.hn + (size_t) (T - 1) * E, E, rows);
            I.mm(I.head, I.logits);
        }
        I.stage("head", s);
        I.prof_flush(T);
        if (verify) pending_ = T;
        else pos_ += T;
        return true;
    } catch (const std::exception& e) {
        err = std::string("dense model: ") + e.what();
        return false;
    }
}

bool DenseModel::commit(int n, std::string& err) {
    Impl& I = *impl_;
    if (pending_ == 0 || n < 1 || n > pending_) { err = "dense model: commit without a matching verify pass"; return false; }
    const int C = cfg_.conv_channels(), KH = cfg_.ssm_k_heads, VH = cfg_.ssm_v_heads;
    try {
        for (Layer& L : I.layers) {
            if (L.attn) continue;
            q35_conv_seq(L.conv_state, L.st_raw, C, L.conv_w, I.conv_dummy, n, C, true, I.stream);
            q35_gdn_seq(L.state, L.st_conv, C, L.st_ab, L.st_ab + VH, 2 * VH, I.o, n, KH, VH, true, I.stream);
        }
    } catch (const std::exception& e) {
        err = std::string("dense model: ") + e.what();
        return false;
    }
    pos_ += n;
    pending_ = 0;
    return true;
}

bool DenseModel::mtp_forward(const int32_t* toks, const float* hidden, int64_t p0, int T, bool all_logits,
                             std::string& err) {
    Impl& I = *impl_;
    Layer& M = I.mtp;
    if (!M.attn) { err = "dense model: no MTP block"; return false; }
    if (T < 1 || T > I.chunk || p0 + T > max_context_) { err = "dense model: MTP rows outside the context"; return false; }
    if (all_logits && T > kMaxCols) { err = "dense model: MTP logits of every row up to 8 rows"; return false; }
    cudaStream_t s = I.stream;
    const int E = cfg_.n_embd;
    const float eps = cfg_.rms_eps;
    try {
        I.mark(s);
        // [enorm(emb(tok)) | hnorm(h)] -> eh_proj
        for (int t = 0; t < T; ++t)
            dequant_f32(I.embd.type, I.embd.dev, toks[t], 1, E, I.y + (size_t) t * E, s);
        float* cat = I.mtp_cat;
        q35_rms_rows(I.y, E, M.enorm, I.xn, nullptr, T, E, eps, s);
        cudaMemcpy2DAsync(cat, (size_t) 2 * E * 4, I.xn, (size_t) E * 4, (size_t) E * 4, T, cudaMemcpyDeviceToDevice, s);
        q35_rms_rows(hidden, E, M.hnorm, I.xn, nullptr, T, E, eps, s);
        cudaMemcpy2DAsync(cat + E, (size_t) 2 * E * 4, I.xn, (size_t) E * 4, (size_t) E * 4, T, cudaMemcpyDeviceToDevice, s);
        I.prep(cat, 2 * E, T);
        float* x = T <= kMaxCols ? I.mtp_x : I.x;   // (a prompt's rows reuse the trunk's buffer: the trunk is done)
        I.mm(M.eh_proj, x);
        q35_rms_rows(x, E, M.attn_norm, I.xn, nullptr, T, E, eps, s);
        I.prep(I.xn, E, T);
        attn_mixer(I, cfg_, M, p0, T, (int) max_context_);
        q35_add(x, I.mix, (int64_t) T * E, s);
        ffn(I, cfg_, M, x, T);
        q35_rms_rows(x, E, M.head_norm, I.mtp_h, nullptr, T, E, eps, s);
        const int rows = all_logits ? T : 1;
        I.prep(all_logits ? I.mtp_h : I.mtp_h + (size_t) (T - 1) * E, E, rows);
        I.mm(I.head_draft, I.mtp_logits);
        return true;
    } catch (const std::exception& e) {
        err = std::string("dense model: ") + e.what();
        return false;
    }
}

bool DenseModel::has_dflash() const { return impl_->df.on; }
int DenseModel::dflash_max() const { return impl_->df.on ? impl_->df.block - 1 : 0; }

bool DenseModel::dflash_load(const std::string& path, std::string& err) {
    Impl& I = *impl_;
    Dflash& F = I.df;
    try {
        I.df_file.reset(new strata::GgufFile(path));
        const strata::GgufFile& G = *I.df_file;
        const auto* arch = G.get("general.architecture");
        if (!arch || arch->s != "qwen35-dflash-draft") {
            err = "dflash: " + path + " is not a qwen35-dflash-draft GGUF";
            return false;
        }
        const std::string a = arch->s + ".";
        auto num = [&](const std::string& k, double def) -> double {
            const auto* v = G.get(a + k);
            return v ? v->num() : def;
        };
        const int E = cfg_.n_embd;
        if ((int) num("embedding_length", 0) != E || (int) num("vocab_size", cfg_.n_vocab) != cfg_.n_vocab) {
            err = "dflash: the drafter's hidden size / vocabulary do not match the model";
            return false;
        }
        F.n_layer = (int) num("block_count", 0);
        F.n_head = (int) num("attention.head_count", 0);
        F.n_kv = (int) num("attention.head_count_kv", 0);
        F.block = (int) num("dflash.block_size", 8);
        F.taps = (int) num("dflash.dflash2.conv_kernel_size", 0);
        F.group = (int) num("dflash.dflash2.conv_group_size", 16);
        F.rank = (int) num("dflash.dflash2.selector_rank", 0);
        F.top_k = (int) num("dflash.dflash2.selector_top_k", 16);
        F.mask_tok = (int) num("dflash.mask_token_id", -1);
        F.eps = (float) num("attention.layer_norm_rms_epsilon", 1e-6);
        F.theta = (float) num("rope.freq_base", 1e7);
        F.window = (int) num("attention.sliding_window", 2048);
        if (const char* e = std::getenv("STRATA_DENSE_DFLASH_CTX")) F.window = std::max(64, std::atoi(e));
        F.n_cand = std::min(cfg_.n_vocab, (int) num("dflash.candidate_vocab", 248077));
        if (const auto* tl = G.get(a + "dflash.target_layer_ids"))
            for (const auto& it : tl->items) F.target_layers.push_back((int) it.u);
        if (F.n_layer < 1 || F.n_head < 1 || F.n_kv < 1 || F.taps != 2 || F.rank < 32 || F.rank > 1024 ||
            F.rank % 32 || F.top_k < 1 || F.top_k > 32 || F.mask_tok < 0 || F.target_layers.empty() ||
            F.block < 2 || F.block > 9 || (int) num("attention.key_length", 128) != 128) {
            err = "dflash: unsupported drafter geometry (DFlash2 with a selector, head size 128, block 2..9)";
            return false;
        }
        for (int l : F.target_layers)
            if (l < 0 || l >= cfg_.n_layer) { err = "dflash: a target layer outside the model"; return false; }
        const int NF = (int) F.target_layers.size();
        uint64_t total = 0;
        auto find = [&](const std::string& n) -> const strata::TensorInfo* { return G.find(n); };
        auto f32 = [&](const std::string& n, uint64_t want, float** out) {
            const auto* t = find(n);
            if (!t) { err = "dflash: missing " + n; return false; }
            Found f{&G, t};
            return I.upload_floats(&f, n, want, out, err);
        };
        // an F16 / BF16 / F32 matrix -> Q8_0 on the device (the GEMVs and GEMMs take Q8_0); a quantized one as is
        auto mat = [&](const std::string& n, int n_in, int n_out, QMat& out) {
            const auto* t = find(n);
            if (!t || t->shape.size() != 2 || t->shape[0] != (uint64_t) n_in || t->shape[1] != (uint64_t) n_out) {
                err = "dflash: " + n + " missing or not " + std::to_string(n_in) + " x " + std::to_string(n_out);
                return false;
            }
            if (t->type != kF16) {
                Found f{&G, t};
                return I.upload_mat(&f, n, n_in, n_out, out, total, err);
            }
            const std::vector<uint8_t> q = f16_to_q8_0((const uint16_t*) G.tensor_data(*t), n_in, n_out);
            void* dev = nullptr;
            if (!I.alloc(&dev, q.size(), err, n.c_str())) return false;
            cudaMemcpy(dev, q.data(), q.size(), cudaMemcpyHostToDevice);
            out = QMat{dev, 8, n_in, n_out};
            total += q.size();
            if (!I.pack(out, true, err)) return false;
            return true;
        };
        const int qd = F.n_head * 128, kvd = F.n_kv * 128, cdyn = 2 * F.taps * (E / F.group);
        if (!mat("dflash.fc.weight", NF * E, E, F.fc) || !mat("dflash.selector.hproj.weight", E, F.rank, F.hproj) ||
            !f32("dflash.hidden_norm.weight", E, &F.hidden_norm) || !f32("output_norm.weight", E, &F.out_norm))
            return false;
        for (const char* n : {"dflash.selector.pred_cb", "dflash.selector.succ_cb"}) {
            const auto* t = find(n);
            if (!t || t->type != kF16 || t->shape.size() != 2 || t->shape[0] != (uint64_t) F.rank ||
                t->shape[1] != (uint64_t) cfg_.n_vocab) {
                err = std::string("dflash: ") + n + " missing or not F16 " + std::to_string(F.rank) + " x vocabulary";
                return false;
            }
            uint16_t*& dst = std::string(n).find("pred") != std::string::npos ? F.pred : F.succ;
            const size_t b = (size_t) F.rank * cfg_.n_vocab * 2;
            if (!I.alloc(&dst, b, err, n)) return false;
            cudaMemcpy(dst, G.tensor_data(*t), b, cudaMemcpyHostToDevice);
            total += b;
        }
        F.layers.assign((size_t) F.n_layer, DfLayer{});
        for (int il = 0; il < F.n_layer; ++il) {
            DfLayer& L = F.layers[(size_t) il];
            const std::string P = "blk." + std::to_string(il) + ".";
            if (!f32(P + "attn_norm.weight", E, &L.attn_norm) || !f32(P + "ffn_norm.weight", E, &L.ffn_norm) ||
                !f32(P + "attn_q_norm.weight", 128, &L.q_norm) || !f32(P + "attn_k_norm.weight", 128, &L.k_norm) ||
                !f32(P + "attn_conv.base", (uint64_t) 2 * F.taps * E, &L.attn_base) ||
                !f32(P + "ffn_conv.base", (uint64_t) 2 * F.taps * E, &L.ffn_base) ||
                !mat(P + "attn_q.weight", E, qd, L.q) || !mat(P + "attn_k.weight", E, kvd, L.k) ||
                !mat(P + "attn_v.weight", E, kvd, L.v) || !mat(P + "attn_output.weight", qd, E, L.o) ||
                !mat(P + "attn_conv.proj.weight", E, cdyn, L.attn_conv) ||
                !mat(P + "ffn_conv.proj.weight", E, cdyn, L.ffn_conv) ||
                !mat(P + "ffn_gate.weight", E, cfg_.n_ff, L.gate) || !mat(P + "ffn_up.weight", E, cfg_.n_ff, L.up) ||
                !mat(P + "ffn_down.weight", cfg_.n_ff, E, L.down))
                return false;
            const uint64_t rb = (uint64_t) F.window * kvd * 2;
            if (!I.alloc(&L.Kr, rb, err, "dflash ring") || !I.alloc(&L.Vr, rb, err, "dflash ring")) return false;
            total += 2 * rb;
        }
        const uint64_t R = (uint64_t) I.chunk, W = (uint64_t) F.block;
        if (!I.alloc(&F.ring_pos, (uint64_t) F.window * 8, err, "dflash ring") ||
            !I.alloc(&F.feat, R * NF * E * 4, err, "dflash features") || !I.alloc(&F.h, W * E * 4, err, "dflash") ||
            !I.alloc(&F.dyn, W * cdyn * 4, err, "dflash") || !I.alloc(&F.a, W * E * 4, err, "dflash") ||
            !I.alloc(&F.hf, W * E * 4, err, "dflash") ||
            !I.alloc(&F.logits, W * cfg_.n_vocab * 4, err, "dflash logits") ||
            !I.alloc(&F.gate, W * F.rank * 4, err, "dflash") ||
            !I.alloc(&F.cand_v, W * F.top_k * 4, err, "dflash") || !I.alloc(&F.cand_i, W * F.top_k * 4, err, "dflash") ||
            !I.alloc(&F.tok, W * 4, err, "dflash") || !I.alloc(&F.conf, W * 4, err, "dflash") ||
            !I.alloc(&F.topk_scratch, (uint64_t) q35_df_topk_scratch((int) W, F.top_k), err, "dflash"))
            return false;
        cudaMemset(F.ring_pos, 0xff, (size_t) F.window * 8);   // -1: no position
        if (cudaDeviceSynchronize() != cudaSuccess) { err = "dflash: upload failed"; return false; }
        F.on = true;
        std::fprintf(stderr, "strata-dense: DFlash2 drafter %s: %d layers, block %d, %.2f GiB (Q8_0), window %d, "
                     "target layers", path.c_str(), F.n_layer, F.block, total / 1073741824.0, F.window);
        for (int l : F.target_layers) std::fprintf(stderr, " %d", l);
        std::fprintf(stderr, "\n");
        return true;
    } catch (const std::exception& e) {
        err = std::string("dflash: ") + e.what();
        return false;
    }
}

bool DenseModel::dflash_inject(int n, int64_t p0, std::string& err) {
    Impl& I = *impl_;
    Dflash& F = I.df;
    if (!F.on || n <= 0) return true;
    cudaStream_t s = I.stream;
    const int E = cfg_.n_embd, NF = (int) F.target_layers.size(), kvd = F.n_kv * 128;
    try {
        // c = rms(fc . features, hidden_norm); per layer K = rope(rms_head(Wk c) * k_norm) at the row's position, V = Wv c
        I.prep(F.feat, NF * E, n);
        I.mm(F.fc, I.mix);
        q35_rms_rows(I.mix, E, F.hidden_norm, I.xn, nullptr, n, E, F.eps, s);
        I.prep(I.xn, E, n);
        for (DfLayer& L : F.layers) {
            I.mm(L.k, I.kcur);
            I.mm(L.v, I.vcur);
            q35_df_head_norm_rope(I.kcur, n, F.n_kv, L.k_norm, F.eps, p0, F.theta, s);
            // rows beyond the window would overwrite each other's slots: keep the last `window`
            const int skip = std::max(0, n - F.window);
            q35_df_ring_store(I.kcur + (size_t) skip * kvd, I.vcur + (size_t) skip * kvd, n - skip, kvd, p0 + skip,
                              F.window, L.Kr, L.Vr, F.ring_pos, s);
        }
        return true;
    } catch (const std::exception& e) {
        err = std::string("dflash: ") + e.what();
        return false;
    }
}

bool DenseModel::dflash_draft(int32_t anchor, int K, std::vector<int32_t>& out, std::string& err) {
    Impl& I = *impl_;
    Dflash& F = I.df;
    out.clear();
    if (!F.on || K < 1) return true;
    K = std::min(K, F.block - 1);
    const int W = K + 1, E = cfg_.n_embd, kvd = F.n_kv * 128, qd = F.n_head * 128;
    const int cdyn = 2 * F.taps * (E / F.group);
    const int64_t P0 = pos_;
    cudaStream_t s = I.stream;
    try {
        // h = emb([anchor, mask x K])
        for (int t = 0; t < W; ++t)
            dequant_f32(I.embd.type, I.embd.dev, t == 0 ? anchor : F.mask_tok, 1, E, F.h + (size_t) t * E, s);
        for (DfLayer& L : F.layers) {
            // attention, wrapped in the attention conv
            q35_rms_rows(F.h, E, L.attn_norm, I.xn, nullptr, W, E, F.eps, s);
            I.prep(I.xn, E, W);
            I.mm(L.attn_conv, F.dyn);
            q35_df_conv(I.xn, F.dyn, cdyn, L.attn_base, 0, W, E, F.taps, F.group, F.a, s);
            I.prep(F.a, E, W);
            I.mm(L.q, I.q_full);
            I.mm(L.k, I.kcur);
            I.mm(L.v, I.vcur);
            q35_df_head_norm_rope(I.q_full, W, F.n_head, L.q_norm, F.eps, P0, F.theta, s);
            q35_df_head_norm_rope(I.kcur, W, F.n_kv, L.k_norm, F.eps, P0, F.theta, s);
            q35_df_attn(I.q_full, L.Kr, L.Vr, F.ring_pos, I.kcur, I.vcur, W, F.n_head, F.n_kv, P0, F.window,
                        F.window, I.attn, s);
            I.prep(I.attn, qd, W);
            I.mm(L.o, I.mix);
            q35_df_conv(I.mix, F.dyn, cdyn, L.attn_base, 1, W, E, F.taps, F.group, F.a, s);
            q35_add(F.h, F.a, (int64_t) W * E, s);
            // the MLP, wrapped in the MLP conv
            q35_rms_rows(F.h, E, L.ffn_norm, I.xn, nullptr, W, E, F.eps, s);
            I.prep(I.xn, E, W);
            I.mm(L.ffn_conv, F.dyn);
            q35_df_conv(I.xn, F.dyn, cdyn, L.ffn_base, 0, W, E, F.taps, F.group, F.a, s);
            I.prep(F.a, E, W);
            I.mm(L.gate, I.ffn_g);
            I.mm(L.up, I.ffn_u);
            dense_swiglu(I.ffn_g, I.ffn_u, I.ffn_h, (int64_t) W * cfg_.n_ff, s);
            I.prep(I.ffn_h, cfg_.n_ff, W);
            I.mm(L.down, I.mix);
            q35_df_conv(I.mix, F.dyn, cdyn, L.ffn_base, 1, W, E, F.taps, F.group, F.a, s);
            q35_add(F.h, F.a, (int64_t) W * E, s);
        }
        (void) kvd;
        q35_rms_rows(F.h, E, F.out_norm, F.hf, nullptr, W, E, F.eps, s);
        // the mask rows' candidates through the target's head, their gates, the walk
        I.prep(F.hf + E, E, K);
        I.mm(I.head_draft, F.logits);
        q35_df_topk(F.logits, I.head_draft.n_out, std::min(F.n_cand, I.head_draft.n_out), K, F.top_k, F.cand_i,
                    F.cand_v, F.topk_scratch, s);
        I.mm(F.hproj, F.gate);
        q35_df_walk(F.cand_i, F.cand_v, F.gate, F.top_k, F.rank, F.pred, F.succ, anchor, K, F.tok, F.conf, s);
        out.resize((size_t) K);
        cudaMemcpyAsync(out.data(), F.tok, (size_t) K * 4, cudaMemcpyDeviceToHost, s);
        if (cudaStreamSynchronize(s) != cudaSuccess) { err = "dflash: draft failed"; return false; }
        return true;
    } catch (const std::exception& e) {
        err = std::string("dflash: ") + e.what();
        return false;
    }
}

uint64_t DenseModel::ckpt_bytes() const {
    const Impl& I = *impl_;
    const uint64_t sb = (uint64_t) cfg_.ssm_state * cfg_.ssm_v_heads * cfg_.ssm_state * 4;
    const uint64_t cb = (uint64_t) cfg_.conv_channels() * (cfg_.ssm_d_conv - 1) * 4;
    uint64_t n = 0;
    for (const Layer& L : I.layers)
        if (!L.attn) n += sb + cb;
    if (I.df.on) n += (uint64_t) I.df.n_layer * 2 * I.df.window * I.df.n_kv * 128 * 2 + (uint64_t) I.df.window * 8;
    return n;
}

namespace {
// every buffer of the recurrent state, in a fixed order: (device pointer, bytes)
std::vector<std::pair<void*, uint64_t>> ckpt_parts(DenseModel::Impl& I, const DenseConfig& c) {
    std::vector<std::pair<void*, uint64_t>> v;
    const uint64_t sb = (uint64_t) c.ssm_state * c.ssm_v_heads * c.ssm_state * 4;
    const uint64_t cb = (uint64_t) c.conv_channels() * (c.ssm_d_conv - 1) * 4;
    for (Layer& L : I.layers)
        if (!L.attn) { v.push_back({L.state, sb}); v.push_back({L.conv_state, cb}); }
    if (I.df.on) {
        const uint64_t rb = (uint64_t) I.df.window * I.df.n_kv * 128 * 2;
        for (DfLayer& L : I.df.layers) { v.push_back({L.Kr, rb}); v.push_back({L.Vr, rb}); }
        v.push_back({I.df.ring_pos, (uint64_t) I.df.window * 8});
    }
    return v;
}
}  // namespace

bool DenseModel::ckpt_save(int slot, std::string& err) {
    Impl& I = *impl_;
    if (slot < 0 || pending_) { err = "dense model: checkpoint slot / pending verify"; return false; }
    if ((int) I.ckpt.size() <= slot) I.ckpt.resize((size_t) slot + 1, nullptr);
    if (!I.ckpt[(size_t) slot] &&
        cudaHostAlloc((void**) &I.ckpt[(size_t) slot], ckpt_bytes(), cudaHostAllocDefault) != cudaSuccess) {
        I.ckpt[(size_t) slot] = nullptr;
        err = "dense model: no pinned memory for a checkpoint";
        return false;
    }
    uint8_t* h = I.ckpt[(size_t) slot];
    for (auto& [p, b] : ckpt_parts(I, cfg_)) {
        cudaMemcpyAsync(h, p, b, cudaMemcpyDeviceToHost, I.stream);
        h += b;
    }
    return sync(err);
}

bool DenseModel::ckpt_restore(int slot, int64_t pos, std::string& err) {
    Impl& I = *impl_;
    if (slot < 0 || slot >= (int) I.ckpt.size() || !I.ckpt[(size_t) slot] || pos < 0 || pos > max_context_) {
        err = "dense model: no such checkpoint";
        return false;
    }
    const uint8_t* h = I.ckpt[(size_t) slot];
    for (auto& [p, b] : ckpt_parts(I, cfg_)) {
        cudaMemcpyAsync(p, h, b, cudaMemcpyHostToDevice, I.stream);
        h += b;
    }
    pos_ = pos;
    pending_ = 0;
    return sync(err);
}

}  // namespace strata::core
