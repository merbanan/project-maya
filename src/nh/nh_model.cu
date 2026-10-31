// src/nh/nh_model.cu - maya-nh's model (nh_model.hpp): loading and the forward.
//
// A block's forward, as transformers' NemotronHBlock (tools/nh/ref.py):  x = rms(h) * w;  then by the block's type
//   M  Mamba-2: zxbcdt = x Win;  xbc = silu(conv(xbc));  y = scan(x, B, C, dt);  h += gated_rms(y, z) Wout
//   A  attention: q, k, v = x Wq, x Wk, x Wv;  K, V -> the BF16 cache;  causal GQA (no position encoding);  h += o Wo
//   E  MoE: the router (F32 logits, sigmoid, + bias, top 22, renormalised, x 5);  lat = x Wfc1;  the routed experts on
//      lat (down(relu(up lat)^2), weighted);  h += (sum) Wfc2 + down_s(relu(up_s x)^2)
// then the final norm and the head.  The matrices are EXL3 (maya-mm3's kernels: mv for one token, gemm_rows for
// more); the attention is maya-mm3's sparse attention with every block kept.
#include "nh_model.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <numeric>
#include <sstream>

#include "mm3_exl3.hpp"
#include "mm3_kernels.cuh"
#include "mm3_pack.hpp"
#include "nh_cpu.hpp"
#include "nh_kernels.cuh"

namespace nh {
namespace x3 = strata::mm3::exl3;
namespace mk = mm3::k;

#define CK(call)                                                                       \
    do {                                                                               \
        const cudaError_t e_ = (call);                                                 \
        if (e_ != cudaSuccess) {                                                       \
            err = std::string(#call) + ": " + cudaGetErrorString(e_);                  \
            return false;                                                              \
        }                                                                              \
    } while (0)
#define CB(call)                                                                       \
    do {                                                                               \
        const cublasStatus_t b_ = (call);                                              \
        if (b_ != CUBLAS_STATUS_SUCCESS) {                                             \
            err = std::string(#call) + ": cublas status " + std::to_string((int) b_);  \
            return false;                                                              \
        }                                                                              \
    } while (0)

namespace {

constexpr size_t kAlign = 256;
constexpr int kBlk = 64;        // the attention's blocks (every one kept): maya-mm3's tile
constexpr int kSplits = 32;     // a decode token's attention in this many runs of blocks (mm3's kCombMaxS)
constexpr int kMvRows = 8;      // (a forward of this many tokens or fewer: the token path's GEMVs, mv's columns)
size_t up(size_t n) { return (n + kAlign - 1) / kAlign * kAlign; }
double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

const std::string P = "language_model.backbone.";

struct Geo {
    int dim = 0, vocab = 0, n_layers = 0;
    std::string types;
    int mh = 0, mp = 0, mn = 0, mg = 0, conv_k = 4;   // Mamba: heads, head dim, state, groups, conv taps
    int hq = 0, hkv = 0, hd = 0;
    int n_exp = 0, k_exp = 0, ff = 0, lat = 0, ff_sh = 0;
    float scale = 1.0f, eps = 1e-5f;
    int bos = 1;
    std::vector<int> eos;
    int k_up = 4, k_down = 4;
    int d_in() const { return mh * mp; }
    int conv_dim() const { return d_in() + 2 * mg * mn; }
    int proj() const { return d_in() + conv_dim() + mh; }

    bool read(const mm3::Pack& pk, std::string& err) {
        dim = (int) pk.num("hidden_size");
        vocab = (int) pk.num("vocab_size");
        types = pk.str("layers_block_type");
        n_layers = (int) types.size();
        mh = (int) pk.num("mamba_num_heads");
        mp = (int) pk.num("mamba_head_dim");
        mn = (int) pk.num("ssm_state_size");
        mg = (int) pk.num("n_groups");
        conv_k = (int) pk.num("conv_kernel", 4);
        hq = (int) pk.num("num_attention_heads");
        hkv = (int) pk.num("num_key_value_heads");
        hd = (int) pk.num("head_dim");
        n_exp = (int) pk.num("n_routed_experts");
        k_exp = (int) pk.num("num_experts_per_tok");
        ff = (int) pk.num("moe_intermediate_size");
        lat = (int) pk.num("moe_latent_size");
        ff_sh = (int) pk.num("moe_shared_expert_intermediate_size");
        scale = (float) pk.num("routed_scaling_factor", 1.0);
        eps = (float) pk.num("layer_norm_epsilon", 1e-5);
        bos = (int) pk.num("bos_token_id", 1);
        eos = pk.ints("eos_token_id");
        const std::vector<int> kb = pk.ints("exl3.experts");
        if (kb.size() == 2) {
            k_up = kb[0];
            k_down = kb[1];
        }
        if (dim <= 0 || n_layers <= 0 || types.find_first_not_of("MAE") != std::string::npos) {
            err = "nh.txt: no hidden_size / layers_block_type of M, A, E";
            return false;
        }
        if (conv_k != 4 || mp != 64 || mn != 128 || mh % mg != 0) {
            err = "nh.txt: a Mamba geometry the kernels do not cover (conv 4, head_dim 64, state 128)";
            return false;
        }
        if (hd != 128 || hq != 16 * hkv) {
            err = "nh.txt: attention heads the kernels do not cover (head_dim 128, 16 query heads a KV head)";
            return false;
        }
        if (k_exp > 32 || n_exp > 1024) {
            err = "nh.txt: routing the kernels do not cover (top 32 of 1024 at most)";
            return false;
        }
        return true;
    }
};

struct Layer {
    char type = 'M';
    float* norm = nullptr;
    // M
    x3::Mat in, out;
    float *conv_w = nullptr, *conv_b = nullptr, *A = nullptr, *D = nullptr, *dt_bias = nullptr, *gnorm = nullptr;
    float *tail = nullptr, *ssm = nullptr;   // the sequence's state
    float *tail_s = nullptr, *ssm_s = nullptr;   // (snapshot)
    // A
    x3::Mat q, k, v, o;
    uint16_t *kc = nullptr, *vc = nullptr;
    // E
    x3::Mat fc1, fc2, su, sd;
    float *gate = nullptr, *bias = nullptr;
    std::vector<int> res_slot;   // expert -> its blob in res_base (-1: not in VRAM)
    uint8_t* res_base = nullptr;
    int n_res = 0;
    std::vector<x3::Mat> res_up, res_down;   // (by expert: the resident ones' matrices)
    std::vector<const uint8_t*> host_blob;   // expert -> its assembled blob in pinned memory (null: none)
};

}  // namespace

struct Model::Impl {
    Options o;
    mm3::Pack pk;
    Geo g;
    Stats st;
    std::vector<Layer> L;
    std::vector<void*> owned;
    x3::Mat head;
    float* norm_f = nullptr;
    int dev = 0;
    cudaStream_t s = nullptr;
    cublasHandle_t blas = nullptr;
    int cur = 0;              // the state's position
    int snap = -1;            // the snapshot's position
    long long fwd = 0;
    std::vector<unsigned> prof;
    // the expert blob: [up trellis | down trellis | up suh | up svh | down suh | down svh], each part 256-aligned
    size_t blob = 0, o_ut = 0, o_dt = 0, o_us = 0, o_uv = 0, o_ds = 0, o_dv = 0;
    // staging
    uint8_t* stg = nullptr;
    int n_slots = 0;
    std::vector<long long> slot_owner, slot_used;
    std::vector<x3::Mat> slot_up, slot_down;
    uint8_t* pin = nullptr;   // a pinned blob (a miss assembled from the files)
    // scratch (chunk rows)
    float *h = nullptr, *xn = nullptr, *zx = nullptr, *xbc = nullptr, *y = nullptr, *tmp = nullptr;
    float *q = nullptr, *kk = nullptr, *vv = nullptr, *att = nullptr, *part = nullptr;
    float *logit = nullptr, *w = nullptr, *lat = nullptr, *lacc = nullptr, *su = nullptr, *eh = nullptr, *ey = nullptr;
    float* lg = nullptr;
    uint16_t *xn16 = nullptr, *y16 = nullptr, *att16 = nullptr, *su16 = nullptr, *lat16 = nullptr, *eh16 = nullptr;
    int *pos = nullptr, *top = nullptr, *kept = nullptr, *rowidx = nullptr, *invrow = nullptr, *ident = nullptr;
    float* emb_h = nullptr;   // pinned [chunk][dim]
    // the CPU lane (a few tokens' experts not in VRAM): the tokens' latent inputs and weights back on the host, its
    // part's way to the device, the GPU experts' weights picked out of each token's
    std::unique_ptr<CpuLane> lane;
    float *lat_h = nullptr, *w_h = nullptr, *cpu_h = nullptr, *wsel_h = nullptr;
    float *cpu_d = nullptr, *wsel_d = nullptr;
    int* top_h = nullptr;     // pinned [chunk][k]
    float* lg_h = nullptr;    // pinned [rows][vocab]
    int head_rows = 0;
    int nblk_max = 0;

    ~Impl() {
        if (dev >= 0) cudaSetDevice(dev);
        if (s) cudaStreamSynchronize(s);
        for (void* p : owned) cudaFree(p);
        for (Layer& l : L)
            for (const uint8_t* b : l.host_blob)
                (void) b;   // (the pool: one allocation a layer, freed below)
        for (void* p : pinned) cudaFreeHost(p);
        if (blas) cublasDestroy(blas);
        if (s) cudaStreamDestroy(s);
    }
    std::vector<void*> pinned;

    template <class T>
    bool dalloc(T*& p, size_t n, std::string& err) {
        void* d = nullptr;
        if (cudaMalloc(&d, std::max<size_t>(n * sizeof(T), 4)) != cudaSuccess) {
            err = "out of device memory (" + std::to_string(n * sizeof(T) >> 20) + " MB)";
            return false;
        }
        owned.push_back(d);
        p = (T*) d;
        return true;
    }
    template <class T>
    bool halloc(T*& p, size_t n, std::string& err) {
        void* d = nullptr;
        if (cudaHostAlloc(&d, std::max<size_t>(n * sizeof(T), 4), cudaHostAllocDefault) != cudaSuccess) {
            err = "out of pinned host memory (" + std::to_string(n * sizeof(T) >> 20) + " MB)";
            return false;
        }
        pinned.push_back(d);
        p = (T*) d;
        return true;
    }

    static bool kcpu_ok() {
        return strata::kernels::cpu::exl3_cpu_available();
    }

    // ---- loading
    x3::Mat upload_mat(const std::string& pre, std::string& err) {
        x3::Mat m;
        const mm3::Tensor* tr = pk.get(pre + ".trellis");
        const mm3::Tensor* su_ = pk.get(pre + ".suh");
        const mm3::Tensor* sv = pk.get(pre + ".svh");
        if (!tr || !su_ || !sv || !pk.get(pre + ".mul1") || tr->shape.size() != 3) {
            err = pre + ": not an EXL3 (mul1) matrix";
            return m;
        }
        uint8_t* d = nullptr;
        const size_t a = up(tr->bytes), b = up(su_->bytes);
        if (!dalloc(d, a + b + sv->bytes, err)) return m;
        cudaMemcpy(d, pk.data(*tr), tr->bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d + a, pk.data(*su_), su_->bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d + a + b, pk.data(*sv), sv->bytes, cudaMemcpyHostToDevice);
        m.trellis = (const uint32_t*) d;
        m.suh = (const uint16_t*) (d + a);
        m.svh = (const uint16_t*) (d + a + b);
        m.k = (int) (su_->bytes / 2);
        m.n = (int) (sv->bytes / 2);
        m.k2 = (int) tr->shape[2] / 8;
        m.cb = 2;
        if (!x3::supported(m)) err = pre + ": shape / bits not covered by the EXL3 kernels";
        return m;
    }
    // a float vector (F32 / F16 / BF16 as stored) on the device as F32; fn (given) maps each value
    float* upload_f32(const std::string& name, size_t want, std::string& err, float (*fn)(float) = nullptr) {
        const mm3::Tensor* t = pk.get(name);
        if (!t) {
            err = name + ": missing";
            return nullptr;
        }
        const size_t es = t->dtype == "F32" ? 4 : 2;
        const size_t n = t->bytes / es;
        if (want != 0 && n != want) {
            err = name + ": " + std::to_string(n) + " values, " + std::to_string(want) + " expected";
            return nullptr;
        }
        std::vector<float> hv(n);
        const uint8_t* d = pk.data(*t);
        for (size_t i = 0; i < n; ++i) {
            float v;
            if (t->dtype == "F32") {
                std::memcpy(&v, d + 4 * i, 4);
            } else {
                uint16_t b;
                std::memcpy(&b, d + 2 * i, 2);
                uint32_t u;
                if (t->dtype == "BF16") {
                    u = (uint32_t) b << 16;
                } else if (t->dtype == "F16") {
                    const uint32_t sg = (uint32_t) (b & 0x8000) << 16, e = (b >> 10) & 0x1f, m = b & 0x3ff;
                    if (e == 0) {
                        float f = (float) m / 1024.0f / 16384.0f;
                        std::memcpy(&u, &f, 4);
                        u |= sg;
                    } else if (e == 31) {
                        u = sg | 0x7f800000u | (m << 13);
                    } else {
                        u = sg | ((e + 112) << 23) | (m << 13);
                    }
                } else {
                    err = name + ": dtype " + t->dtype;
                    return nullptr;
                }
                std::memcpy(&v, &u, 4);
            }
            hv[i] = fn ? fn(v) : v;
        }
        float* dp = nullptr;
        if (!dalloc(dp, n, err)) return nullptr;
        cudaMemcpy(dp, hv.data(), n * 4, cudaMemcpyHostToDevice);
        return dp;
    }

    // one expert's blob (the layout above) assembled from the mapped files into dst
    void assemble(int li, int e, uint8_t* dst) const {
        const std::string m = P + "layers." + std::to_string(li) + ".mixer.experts." + std::to_string(e) + ".";
        const auto put = [&](const std::string& n, size_t off) {
            const mm3::Tensor* t = pk.get(m + n);
            std::memcpy(dst + off, pk.data(*t), t->bytes);
        };
        put("up_proj.trellis", o_ut);
        put("down_proj.trellis", o_dt);
        put("up_proj.suh", o_us);
        put("up_proj.svh", o_uv);
        put("down_proj.suh", o_ds);
        put("down_proj.svh", o_dv);
    }
    void expert_mats(const uint8_t* b, x3::Mat& mu, x3::Mat& md) const {
        mu = x3::Mat{};
        mu.trellis = (const uint32_t*) (b + o_ut);
        mu.suh = (const uint16_t*) (b + o_us);
        mu.svh = (const uint16_t*) (b + o_uv);
        mu.k = g.lat;
        mu.n = g.ff;
        mu.k2 = 2 * g.k_up;
        mu.cb = 2;
        md = x3::Mat{};
        md.trellis = (const uint32_t*) (b + o_dt);
        md.suh = (const uint16_t*) (b + o_ds);
        md.svh = (const uint16_t*) (b + o_dv);
        md.k = g.ff;
        md.n = g.lat;
        md.k2 = 2 * g.k_down;
        md.cb = 2;
    }

    bool load(const Options& opt, std::string& err) {
        const double t0 = now();
        o = opt;
        if (!pk.load(o.pack, err, "nh.txt") || !g.read(pk, err)) return false;
        dev = o.gpu;
        CK(cudaSetDevice(dev));
        CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
        CB(cublasCreate(&blas));
        CB(cublasSetStream(blas, s));
        if (x3::blas_wanted() && !x3::blas_enable(s, 256)) {
            err = "the EXL3 FP16 GEMM's scratch did not allocate";
            return false;
        }
        if (!x3::prepare(s)) {
            err = "the EXL3 kernels' scratch did not allocate";
            return false;
        }
        o.chunk = std::max(1, std::min(o.chunk, o.max_ctx));
        // the expert blob's layout
        {
            const size_t ut = x3::trellis_bytes_k2(g.lat, g.ff, 2 * g.k_up);
            const size_t dt = x3::trellis_bytes_k2(g.ff, g.lat, 2 * g.k_down);
            o_ut = 0;
            o_dt = up(ut);
            o_us = o_dt + up(dt);
            o_uv = o_us + up((size_t) g.lat * 2);
            o_ds = o_uv + up((size_t) g.ff * 2);
            o_dv = o_ds + up((size_t) g.ff * 2);
            blob = o_dv + up((size_t) g.lat * 2);
        }
        // the layers' dense parts
        L.resize((size_t) g.n_layers);
        const auto softneg = [](float v) { return -std::exp(v); };   // A = -exp(A_log)
        for (int li = 0; li < g.n_layers; ++li) {
            Layer& l = L[(size_t) li];
            l.type = g.types[(size_t) li];
            const std::string pre = P + "layers." + std::to_string(li) + ".";
            const std::string m = pre + "mixer.";
            if (!(l.norm = upload_f32(pre + "norm.weight", (size_t) g.dim, err))) return false;
            if (l.type == 'M') {
                l.in = upload_mat(m + "in_proj", err);
                if (!err.empty()) return false;
                l.out = upload_mat(m + "out_proj", err);
                if (!err.empty()) return false;
                if (l.in.n != g.proj() || l.out.k != g.d_in()) {
                    err = m + "in_proj / out_proj: not the geometry's shape";
                    return false;
                }
                if (!(l.conv_w = upload_f32(m + "conv1d.weight", (size_t) g.conv_dim() * 4, err)) ||
                    !(l.conv_b = upload_f32(m + "conv1d.bias", (size_t) g.conv_dim(), err)) ||
                    !(l.A = upload_f32(m + "A_log", (size_t) g.mh, err, softneg)) ||
                    !(l.D = upload_f32(m + "D", (size_t) g.mh, err)) ||
                    !(l.dt_bias = upload_f32(m + "dt_bias", (size_t) g.mh, err)) ||
                    !(l.gnorm = upload_f32(m + "norm.weight", (size_t) g.d_in(), err)))
                    return false;
                if (!dalloc(l.tail, (size_t) g.conv_dim() * k::kConvTail, err) ||
                    !dalloc(l.ssm, (size_t) g.mh * g.mp * g.mn, err) ||
                    !dalloc(l.tail_s, (size_t) g.conv_dim() * k::kConvTail, err) ||
                    !dalloc(l.ssm_s, (size_t) g.mh * g.mp * g.mn, err))
                    return false;
            } else if (l.type == 'A') {
                const std::pair<x3::Mat*, const char*> mats[] = {{&l.q, "q_proj"}, {&l.k, "k_proj"},
                                                                 {&l.v, "v_proj"}, {&l.o, "o_proj"}};
                for (auto [mat, n] : mats) {
                    *mat = upload_mat(m + n, err);
                    if (!err.empty()) return false;
                }
                const size_t rows = (size_t) o.max_ctx * g.hkv * g.hd;
                if (!dalloc(l.kc, rows, err) || !dalloc(l.vc, rows, err)) return false;
            } else {
                const std::pair<x3::Mat*, const char*> mats[] = {{&l.fc1, "fc1_latent_proj"},
                                                                 {&l.fc2, "fc2_latent_proj"},
                                                                 {&l.su, "shared_experts.up_proj"},
                                                                 {&l.sd, "shared_experts.down_proj"}};
                for (auto [mat, n] : mats) {
                    *mat = upload_mat(m + n, err);
                    if (!err.empty()) return false;
                }
                if (!(l.gate = upload_f32(m + "gate.weight", (size_t) g.n_exp * g.dim, err)) ||
                    !(l.bias = upload_f32(m + "gate.e_score_correction_bias", (size_t) g.n_exp, err)))
                    return false;
                for (int e = 0; e < g.n_exp; ++e) {
                    const std::string em = m + "experts." + std::to_string(e) + ".";
                    const mm3::Tensor* ut = pk.get(em + "up_proj.trellis");
                    const mm3::Tensor* dt = pk.get(em + "down_proj.trellis");
                    if (!ut || !dt || ut->shape[2] != 16 * g.k_up || dt->shape[2] != 16 * g.k_down) {
                        err = em + ": not the experts' bits in nh.txt";
                        return false;
                    }
                }
            }
        }
        head = upload_mat("language_model.lm_head", err);
        if (!err.empty()) return false;
        if (!(norm_f = upload_f32(P + "norm_f.weight", (size_t) g.dim, err))) return false;
        // scratch
        const size_t C = (size_t) o.chunk, R = C * g.k_exp;
        nblk_max = (o.max_ctx + kBlk - 1) / kBlk;
        head_rows = 16;
        if (!dalloc(h, C * g.dim, err) || !dalloc(xn, C * g.dim, err) || !dalloc(xn16, C * g.dim, err) ||
            !dalloc(tmp, std::max<size_t>(C * g.dim, (size_t) g.k_exp * g.ff), err) ||
            !dalloc(zx, C * g.proj(), err) || !dalloc(xbc, C * g.conv_dim(), err) ||
            !dalloc(y, C * g.d_in(), err) || !dalloc(y16, C * g.d_in(), err) ||
            !dalloc(q, C * g.hq * g.hd, err) || !dalloc(kk, C * g.hkv * g.hd, err) ||
            !dalloc(vv, C * g.hkv * g.hd, err) || !dalloc(att, C * g.hq * g.hd, err) ||
            !dalloc(att16, C * g.hq * g.hd, err) ||
            !dalloc(part, std::max<size_t>(mk::attn_part_floats(kMvRows, g.hkv, kSplits), 1), err) ||
            !dalloc(kept, (size_t) g.hkv * C * nblk_max, err) || !dalloc(pos, C, err) ||
            !dalloc(logit, C * g.n_exp, err) || !dalloc(w, C * g.k_exp, err) || !dalloc(top, C * g.k_exp, err) ||
            !dalloc(lat, C * g.lat, err) || !dalloc(lacc, C * g.lat, err) || !dalloc(lat16, R * g.lat, err) ||
            !dalloc(su, C * g.ff_sh, err) || !dalloc(su16, C * g.ff_sh, err) ||
            !dalloc(eh, R * g.ff, err) || !dalloc(eh16, R * g.ff, err) || !dalloc(ey, R * g.lat, err) ||
            !dalloc(rowidx, R, err) || !dalloc(invrow, R, err) || !dalloc(ident, (size_t) g.k_exp, err) ||
            !dalloc(lg, (size_t) head_rows * g.vocab, err))
            return false;
        {
            std::vector<int> id((size_t) g.k_exp);
            std::iota(id.begin(), id.end(), 0);
            CK(cudaMemcpy(ident, id.data(), id.size() * 4, cudaMemcpyHostToDevice));
        }
        if (!halloc(emb_h, C * g.dim, err) || !halloc(top_h, C * g.k_exp, err) ||
            !halloc(lg_h, (size_t) head_rows * g.vocab, err) || !halloc(pin, blob, err))
            return false;
        if (!halloc(lat_h, (size_t) kMvRows * g.lat, err) || !halloc(w_h, (size_t) kMvRows * g.k_exp, err) ||
            !halloc(cpu_h, (size_t) kMvRows * g.lat, err) || !halloc(wsel_h, (size_t) kMvRows * g.k_exp, err) ||
            !dalloc(cpu_d, (size_t) kMvRows * g.lat, err) || !dalloc(wsel_d, (size_t) kMvRows * g.k_exp, err))
            return false;
        {
            const int workers = o.cpu_threads == -2 ? CpuLane::auto_workers() : o.cpu_threads;
            if (workers >= 0 && o.pin_pool && kcpu_ok()) {
                lane = std::make_unique<CpuLane>(workers, g.lat, g.ff);
                st.cpu_threads = workers + 1;
            }
        }
        // staging slots
        n_slots = std::max(o.staging, g.k_exp);
        if (!dalloc(stg, (size_t) n_slots * blob, err)) return false;
        slot_owner.assign((size_t) n_slots, -1);
        slot_used.assign((size_t) n_slots, -1);
        slot_up.resize((size_t) n_slots);
        slot_down.resize((size_t) n_slots);
        for (int sl = 0; sl < n_slots; ++sl)
            expert_mats(stg + (size_t) sl * blob, slot_up[(size_t) sl], slot_down[(size_t) sl]);
        // the routed experts: as many in VRAM as fit (the profile's most used first, else by number)
        if (!place_experts(err)) return false;
        reset();
        st.load_s = now() - t0;
        return true;
    }

    bool place_experts(std::string& err) {
        int n_moe = 0;
        for (const Layer& l : L) n_moe += l.type == 'E';
        prof.assign((size_t) g.n_layers * g.n_exp, 0);
        std::vector<unsigned> pr;
        if (!o.profile.empty()) {
            std::ifstream f(o.profile, std::ios::binary);
            pr.resize(prof.size());
            if (!f.read((char*) pr.data(), (std::streamsize) (pr.size() * 4))) {
                err = o.profile + ": not a profile of this model ([layers][experts] uint32)";
                return false;
            }
            prof = pr;
        }
        size_t fr = 0, tot = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        const size_t reserve = (size_t) (o.reserve_gb * (1u << 30));
        const size_t avail = fr > reserve ? fr - reserve : 0;
        int per = (int) std::min<size_t>((size_t) g.n_exp, avail / blob / std::max(1, n_moe));
        if (o.resident_cap >= 0) per = std::min(per, o.resident_cap);
        for (int li = 0; li < g.n_layers; ++li) {
            Layer& l = L[(size_t) li];
            if (l.type != 'E') continue;
            std::vector<int> order((size_t) g.n_exp);
            std::iota(order.begin(), order.end(), 0);
            if (!pr.empty())
                std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
                    return pr[(size_t) li * g.n_exp + a] > pr[(size_t) li * g.n_exp + b];
                });
            l.res_slot.assign((size_t) g.n_exp, -1);
            l.res_up.resize((size_t) g.n_exp);
            l.res_down.resize((size_t) g.n_exp);
            l.n_res = per;
            if (per > 0 && !dalloc(l.res_base, (size_t) per * blob, err)) return false;
            for (int i = 0; i < per; ++i) {
                const int e = order[(size_t) i];
                l.res_slot[(size_t) e] = i;
                assemble(li, e, pin);
                CK(cudaMemcpy(l.res_base + (size_t) i * blob, pin, blob, cudaMemcpyHostToDevice));
                expert_mats(l.res_base + (size_t) i * blob, l.res_up[(size_t) e], l.res_down[(size_t) e]);
            }
            st.resident += per;
            st.experts += g.n_exp;
            // the others in one pinned pool a layer
            l.host_blob.assign((size_t) g.n_exp, nullptr);
            const int rest = g.n_exp - per;
            if (o.pin_pool && rest > 0) {
                uint8_t* pool = nullptr;
                std::string e2;
                if (halloc(pool, (size_t) rest * blob, e2)) {
                    int j = 0;
                    for (int e = 0; e < g.n_exp; ++e)
                        if (l.res_slot[(size_t) e] < 0) {
                            uint8_t* b = pool + (size_t) j++ * blob;
                            assemble(li, e, b);
                            l.host_blob[(size_t) e] = b;
                        }
                    st.pinned_gb += (double) rest * blob / (1u << 30);
                }
            }
        }
        st.vram_gb = (double) st.resident * blob / (1u << 30);
        return true;
    }

    void reset() {
        for (Layer& l : L)
            if (l.type == 'M') {
                cudaMemsetAsync(l.tail, 0, (size_t) g.conv_dim() * k::kConvTail * 4, s);
                cudaMemsetAsync(l.ssm, 0, (size_t) g.mh * g.mp * g.mn * 4, s);
            }
        cur = 0;
    }

    // ---- the forward's parts
    // y [T][n] (=, or += with acc) = x W: T <= kMvRows one mv per token (the weights read once a token... the decode),
    // else gemm_rows over the FP16 rows
    bool mat(const x3::Mat& m, const float* xf, const uint16_t* x16, int T, float* yv, bool acc, std::string& err) {
        if (T <= kMvRows) {
            float* dst = acc ? tmp : yv;
            x3::MvJob j;
            j.m = &m;
            j.xf = xf;
            j.y = dst;
            j.ncol = T;
            j.xf_cs = (size_t) m.k * 4;
            j.y_cs = (size_t) m.n * 4;
            if (!x3::mv(&j, 1, s)) {
                err = "exl3 mv failed (k " + std::to_string(m.k) + ", n " + std::to_string(m.n) + ")";
                return false;
            }
            if (acc) mk::add(yv, tmp, (size_t) T * m.n, s);
            return true;
        }
        x3::gemm_rows(m, x16, m.k, T, yv, m.n, acc ? 1.0f : 0.0f, s);
        return true;
    }

    bool mamba(Layer& l, int T, std::string& err) {
        if (!mat(l.in, xn, xn16, T, zx, false, err)) return false;
        const int P_ = g.proj(), DI = g.d_in(), CD = g.conv_dim();
        k::conv_silu(zx + DI, P_, T, CD, l.conv_w, l.conv_b, l.tail, xbc, s);
        k::ssm_scan(xbc, T, g.mh, g.mp, g.mg, g.mn, zx + DI + CD, P_, l.dt_bias, l.A, l.D, l.ssm, y, s);
        k::gated_norm(y, zx, P_, T, DI, DI / g.mg, l.gnorm, g.eps, y, y16, s);
        return mat(l.out, y, y16, T, h, true, err);
    }

    bool attention(Layer& l, int T, std::string& err) {
        if (!mat(l.q, xn, xn16, T, q, false, err) || !mat(l.k, xn, xn16, T, kk, false, err) ||
            !mat(l.v, xn, xn16, T, vv, false, err))
            return false;
        k::kv_store(kk, vv, pos, T, g.hkv, l.kc, l.vc, s);
        const int nblk = (cur + T - 1) / kBlk + 1;
        k::kept_all(pos, T, g.hkv, nblk, kBlk, kept, s);
        const bool split = T <= kMvRows;
        mk::sparse_attn(q, l.kc, l.vc, kept, pos, T, g.hq, g.hkv, nblk, kBlk, 1.0f / std::sqrt((float) g.hd), att, s,
                        mk::KvQ{}, mk::KvHost{}, split ? kSplits : 1, split ? part : nullptr);
        mk::to_f16(att, T, g.hq * g.hd, g.hq * g.hd, att16, g.hq * g.hd, s);
        return mat(l.o, att, att16, T, h, true, err);
    }

    // the device address of expert e of layer li (its matrices in mu / md), staging it into a free slot if it is not
    // in VRAM; busy: the slots this call holds
    bool locate(int li, int e, std::vector<char>& busy, const x3::Mat*& mu, const x3::Mat*& md, std::string& err) {
        Layer& l = L[(size_t) li];
        if (l.res_slot[(size_t) e] >= 0) {
            mu = &l.res_up[(size_t) e];
            md = &l.res_down[(size_t) e];
            return true;
        }
        const long long key = (long long) li * 1024 + e;
        int at = -1;
        for (int sl = 0; sl < n_slots; ++sl)
            if (slot_owner[(size_t) sl] == key) at = sl;
        if (at < 0) {
            for (int sl = 0; sl < n_slots; ++sl)
                if (!busy[(size_t) sl] && (at < 0 || slot_used[(size_t) sl] < slot_used[(size_t) at])) at = sl;
            if (at < 0) {
                err = "no free staging slot";
                return false;
            }
            const uint8_t* src = l.host_blob[(size_t) e];
            if (src == nullptr) {
                CK(cudaStreamSynchronize(s));   // (the pinned blob is free)
                assemble(li, e, pin);
                src = pin;
            }
            CK(cudaMemcpyAsync(stg + (size_t) at * blob, src, blob, cudaMemcpyHostToDevice, s));
            if (src == pin) CK(cudaStreamSynchronize(s));
            slot_owner[(size_t) at] = key;
            ++st.staged;
        }
        busy[(size_t) at] = 1;
        slot_used[(size_t) at] = fwd;
        mu = &slot_up[(size_t) at];
        md = &slot_down[(size_t) at];
        return true;
    }

    bool moe(Layer& l, int li, int T, std::string& err) {
        const int K = g.k_exp;
        if (T <= 16) {
            mk::rows_dot(l.gate, g.n_exp, g.dim, xn, T, logit, s);
        } else {
            const float one = 1.0f, zero = 0.0f;
            CB(cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, g.n_exp, T, g.dim, &one, l.gate, g.dim, xn, g.dim, &zero,
                           logit, g.n_exp));
        }
        k::route(logit, l.bias, T, g.n_exp, K, g.scale, top, w, s);
        CK(cudaMemcpyAsync(top_h, top, (size_t) T * K * 4, cudaMemcpyDeviceToHost, s));
        // the latent input and the shared expert while the routing comes back
        if (!mat(l.fc1, xn, xn16, T, lat, false, err)) return false;
        if (T <= kMvRows && lane) {
            CK(cudaMemcpyAsync(lat_h, lat, (size_t) T * g.lat * 4, cudaMemcpyDeviceToHost, s));
            CK(cudaMemcpyAsync(w_h, w, (size_t) T * K * 4, cudaMemcpyDeviceToHost, s));
        }
        if (!mat(l.su, xn, xn16, T, su, false, err)) return false;
        k::relu2(su, (size_t) T * g.ff_sh, su, su16, s);
        if (!mat(l.sd, su, su16, T, h, true, err)) return false;
        CK(cudaStreamSynchronize(s));
        for (int i = 0; i < T * K; ++i) ++prof[(size_t) li * g.n_exp + top_h[i]];
        st.profiled += (long) T * K;
        if (T <= kMvRows) {
            if (!experts_tokens(li, T, err)) return false;
        } else if (!experts_rows(li, T, err)) {
            return false;
        }
        // h += (the routed sum) Wfc2
        if (T > kMvRows) mk::to_f16(lacc, T, g.lat, g.lat, lat16, g.lat, s);
        return mat(l.fc2, lacc, lat16, T, h, true, err);
    }

    // a few tokens: each token's experts on mv (up: lat -> eh, relu^2, down: eh -> ey), then the weighted sum; with
    // the CPU lane, the experts neither in VRAM nor in a staging slot computed on the CPU meanwhile (from the pinned
    // pool) and their part added after the GPU's
    bool experts_tokens(int li, int T, std::string& err) {
        Layer& l = L[(size_t) li];
        const int K = g.k_exp;
        for (int t = 0; t < T; ++t) {
            std::vector<char> busy((size_t) n_slots, 0);
            std::vector<const x3::Mat*> mu, md;
            std::vector<CpuExpert> cpu;
            std::vector<int> gslot;   // the GPU's experts' slots among the token's K
            for (int i = 0; i < K; ++i) {
                const int e = top_h[(size_t) t * K + i];
                if (lane && l.res_slot[(size_t) e] < 0 && l.host_blob[(size_t) e] != nullptr &&
                    !in_slot(li, e)) {
                    CpuExpert c;
                    const uint8_t* b = l.host_blob[(size_t) e];
                    c.up = {(const uint32_t*) (b + o_ut), (const uint16_t*) (b + o_us), (const uint16_t*) (b + o_uv),
                            g.lat, g.ff, 2 * g.k_up};
                    c.down = {(const uint32_t*) (b + o_dt), (const uint16_t*) (b + o_ds),
                              (const uint16_t*) (b + o_dv), g.ff, g.lat, 2 * g.k_down};
                    c.w = w_h[(size_t) t * K + i];
                    cpu.push_back(c);
                    continue;
                }
                const x3::Mat *a = nullptr, *d = nullptr;
                if (!locate(li, e, busy, a, d, err)) return false;
                mu.push_back(a);
                md.push_back(d);
                gslot.push_back(i);
            }
            const int ng = (int) gslot.size();
            for (int pass = 0; pass < 2 && ng > 0; ++pass) {
                for (int i0 = 0; i0 < ng; i0 += x3::kMaxMvJobs) {
                    x3::MvJob jobs[x3::kMaxMvJobs];
                    const int n = std::min(x3::kMaxMvJobs, ng - i0);
                    for (int j = 0; j < n; ++j) {
                        const int i = i0 + j;
                        jobs[j].m = pass == 0 ? mu[(size_t) i] : md[(size_t) i];
                        jobs[j].xf = pass == 0 ? lat + (size_t) t * g.lat : eh + (size_t) i * g.ff;
                        jobs[j].y = pass == 0 ? eh + (size_t) i * g.ff : ey + (size_t) i * g.lat;
                    }
                    if (!x3::mv(jobs, n, s)) {
                        err = "exl3 mv (experts) failed";
                        return false;
                    }
                }
                if (pass == 0) k::relu2(eh, (size_t) ng * g.ff, eh, nullptr, s);
            }
            if (cpu.empty()) {
                mk::moe_combine(nullptr, ey, ident, w + (size_t) t * K, nullptr, 1, K, g.lat, 1.0f,
                                lacc + (size_t) t * g.lat, s);
            } else {
                // the GPU's experts' weights (theirs alone, in order), the sum, then the CPU's part added
                for (int j = 0; j < ng; ++j) wsel_h[(size_t) t * K + j] = w_h[(size_t) t * K + gslot[(size_t) j]];
                CK(cudaMemcpyAsync(wsel_d + (size_t) t * K, wsel_h + (size_t) t * K, (size_t) std::max(ng, 1) * 4,
                                   cudaMemcpyHostToDevice, s));
                if (ng > 0)
                    mk::moe_combine(nullptr, ey, ident, wsel_d + (size_t) t * K, nullptr, 1, ng, g.lat, 1.0f,
                                    lacc + (size_t) t * g.lat, s);
                else
                    CK(cudaMemsetAsync(lacc + (size_t) t * g.lat, 0, (size_t) g.lat * 4, s));
                lane->run(cpu, lat_h + (size_t) t * g.lat, cpu_h + (size_t) t * g.lat);
                st.cpu_experts += (long) cpu.size();
                CK(cudaMemcpyAsync(cpu_d + (size_t) t * g.lat, cpu_h + (size_t) t * g.lat, (size_t) g.lat * 4,
                                   cudaMemcpyHostToDevice, s));
                mk::add(lacc + (size_t) t * g.lat, cpu_d + (size_t) t * g.lat, (size_t) g.lat, s);
            }
            if (t + 1 < T) CK(cudaStreamSynchronize(s));   // (eh / ey and the slots again for the next token)
        }
        return true;
    }

    bool in_slot(int li, int e) const {
        const long long key = (long long) li * 1024 + e;
        for (int sl = 0; sl < n_slots; ++sl)
            if (slot_owner[(size_t) sl] == key) return true;
        return false;
    }

    // a prompt chunk: the rows grouped by expert (each expert's rows one GEMM: up, relu^2, down), the resident experts
    // in one pass, the others staged a batch of slots at a time
    bool experts_rows(int li, int T, std::string& err) {
        Layer& l = L[(size_t) li];
        const int K = g.k_exp;
        std::vector<std::vector<int>> rows_of((size_t) g.n_exp);
        for (int t = 0; t < T; ++t)
            for (int i = 0; i < K; ++i) rows_of[(size_t) top_h[(size_t) t * K + i]].push_back(t * K + i);
        std::vector<int> res, stag;
        for (int e = 0; e < g.n_exp; ++e)
            if (!rows_of[(size_t) e].empty()) (l.res_slot[(size_t) e] >= 0 ? res : stag).push_back(e);
        std::vector<std::vector<int>> passes;
        if (!res.empty()) passes.push_back(res);
        for (size_t i = 0; i < stag.size(); i += (size_t) n_slots)
            passes.emplace_back(stag.begin() + (long) i, stag.begin() + (long) std::min(stag.size(), i + n_slots));
        std::vector<int> ridx, inv((size_t) T * K, -1), start((size_t) g.n_exp, 0);
        ridx.reserve((size_t) T * K);
        for (const auto& ps : passes)
            for (int e : ps) {
                start[(size_t) e] = (int) ridx.size();
                for (int ent : rows_of[(size_t) e]) {
                    inv[(size_t) ent] = (int) ridx.size();
                    ridx.push_back(ent / K);
                }
            }
        CK(cudaMemcpyAsync(rowidx, ridx.data(), ridx.size() * 4, cudaMemcpyHostToDevice, s));
        CK(cudaMemcpyAsync(invrow, inv.data(), inv.size() * 4, cudaMemcpyHostToDevice, s));
        x3::rows_f16(lat, g.lat, rowidx, (int) ridx.size(), g.lat, lat16, s);
        for (const auto& ps : passes) {
            std::vector<char> busy((size_t) n_slots, 0);
            int r0 = -1, r1 = 0;
            for (int e : ps) {
                const x3::Mat *mu = nullptr, *md = nullptr;
                if (!locate(li, e, busy, mu, md, err)) return false;
                const int b = start[(size_t) e], n = (int) rows_of[(size_t) e].size();
                if (r0 < 0) r0 = b;
                r1 = b + n;
                x3::gemm_rows(*mu, lat16 + (size_t) b * g.lat, g.lat, n, eh + (size_t) b * g.ff, g.ff, 0.0f, s);
            }
            k::relu2(eh + (size_t) r0 * g.ff, (size_t) (r1 - r0) * g.ff, nullptr, eh16 + (size_t) r0 * g.ff, s);
            for (int e : ps) {
                const x3::Mat *mu = nullptr, *md = nullptr;
                if (!locate(li, e, busy, mu, md, err)) return false;
                const int b = start[(size_t) e], n = (int) rows_of[(size_t) e].size();
                x3::gemm_rows(*md, eh16 + (size_t) b * g.ff, g.ff, n, ey + (size_t) b * g.lat, g.lat, 0.0f, s);
            }
            if (&ps != &passes.back()) CK(cudaStreamSynchronize(s));   // (the slots again in the next pass)
        }
        mk::moe_combine(nullptr, ey, invrow, w, nullptr, T, K, g.lat, 1.0f, lacc, s);
        return true;
    }

    bool chunk(const int* ids, int T, int want, float* logits, std::string& err) {
        // the embeddings (BF16 rows of the mapped file) as F32
        const mm3::Tensor* et = pk.get(P + "embeddings.weight");
        const uint16_t* E = (const uint16_t*) pk.data(*et);
        for (int t = 0; t < T; ++t) {
            if (ids[t] < 0 || ids[t] >= g.vocab) {
                err = "token id " + std::to_string(ids[t]) + " outside the vocabulary";
                return false;
            }
            const uint16_t* r = E + (size_t) ids[t] * g.dim;
            float* d = emb_h + (size_t) t * g.dim;
            for (int i = 0; i < g.dim; ++i) {
                const uint32_t u = (uint32_t) r[i] << 16;
                std::memcpy(d + i, &u, 4);
            }
        }
        CK(cudaMemcpyAsync(h, emb_h, (size_t) T * g.dim * 4, cudaMemcpyHostToDevice, s));
        mk::iota(pos, T, cur, s);
        ++fwd;
        for (int li = 0; li < g.n_layers; ++li) {
            Layer& l = L[(size_t) li];
            mk::rms(h, l.norm, T, g.dim, g.eps, xn, xn16, s);
            const bool ok = l.type == 'M' ? mamba(l, T, err) : l.type == 'A' ? attention(l, T, err) : moe(l, li, T, err);
            if (!ok) return false;
        }
        cur += T;
        if (want > 0) {
            for (int r0 = 0; r0 < want; r0 += head_rows) {
                const int n = std::min(head_rows, want - r0);
                const float* hr = h + (size_t) (T - want + r0) * g.dim;
                mk::rms(hr, norm_f, n, g.dim, g.eps, xn, xn16, s);
                if (!mat(head, xn, xn16, n, lg, false, err)) return false;
                CK(cudaMemcpyAsync(lg_h, lg, (size_t) n * g.vocab * 4, cudaMemcpyDeviceToHost, s));
                CK(cudaStreamSynchronize(s));
                std::memcpy(logits + (size_t) r0 * g.vocab, lg_h, (size_t) n * g.vocab * 4);
            }
        }
        CK(cudaStreamSynchronize(s));
        if (x3::launch_errors() != 0) {
            err = "an EXL3 kernel did not launch (see stderr)";
            return false;
        }
        return true;
    }

    bool forward(const int* ids, int n, int pos0, int want, float* logits, std::string& err) {
        CK(cudaSetDevice(dev));
        if (pos0 == 0 && cur != 0) reset();
        if (pos0 != cur) {
            err = "forward at " + std::to_string(pos0) + ": the state is at " + std::to_string(cur);
            return false;
        }
        if (pos0 + n > o.max_ctx) {
            err = "context full (" + std::to_string(o.max_ctx) + ")";
            return false;
        }
        want = std::min(want, n);
        for (int i = 0; i < n;) {
            const int T = std::min(o.chunk, n - i);
            // this chunk's rows among the call's last `want`: [max(i, n - want), i + T)
            const int first = std::max(i, n - want);
            const int w_here = std::max(0, i + T - first);
            float* dst = logits && w_here > 0 ? logits + (size_t) (first - (n - want)) * g.vocab : nullptr;
            if (!chunk(ids + i, T, dst ? w_here : 0, dst, err)) return false;
            i += T;
        }
        return true;
    }

    bool snapshot(std::string& err) {
        for (Layer& l : L)
            if (l.type == 'M') {
                CK(cudaMemcpyAsync(l.tail_s, l.tail, (size_t) g.conv_dim() * k::kConvTail * 4,
                                   cudaMemcpyDeviceToDevice, s));
                CK(cudaMemcpyAsync(l.ssm_s, l.ssm, (size_t) g.mh * g.mp * g.mn * 4, cudaMemcpyDeviceToDevice, s));
            }
        CK(cudaStreamSynchronize(s));
        snap = cur;
        return true;
    }
    bool restore(std::string& err) {
        if (snap < 0) {
            err = "no snapshot";
            return false;
        }
        for (Layer& l : L)
            if (l.type == 'M') {
                CK(cudaMemcpyAsync(l.tail, l.tail_s, (size_t) g.conv_dim() * k::kConvTail * 4,
                                   cudaMemcpyDeviceToDevice, s));
                CK(cudaMemcpyAsync(l.ssm, l.ssm_s, (size_t) g.mh * g.mp * g.mn * 4, cudaMemcpyDeviceToDevice, s));
            }
        CK(cudaStreamSynchronize(s));
        cur = snap;
        return true;
    }
};

Model::Model() : p(new Impl) {}
Model::~Model() { delete p; }
bool Model::load(const Options& o, std::string& err) { return p->load(o, err); }
bool Model::forward(const int* ids, int n, int pos0, int want, float* logits_out, std::string& err) {
    return p->forward(ids, n, pos0, want, logits_out, err);
}
int Model::pos() const { return p->cur; }
bool Model::snapshot(std::string& err) { return p->snapshot(err); }
int Model::snapshot_pos() const { return p->snap; }
bool Model::restore(std::string& err) { return p->restore(err); }
int Model::vocab() const { return p->g.vocab; }
int Model::max_ctx() const { return p->o.max_ctx; }
int Model::bos() const { return p->g.bos; }
std::vector<int> Model::eos() const { return p->g.eos; }
const Stats& Model::stats() const { return p->st; }
std::vector<std::string> Model::info_lines() const {
    const Impl& m = *p;
    std::vector<std::string> v;
    char b[512];
    std::snprintf(b, sizeof b, "nh: %d layers (%s), dim %d, %d experts top %d, latent %d, context %d", m.g.n_layers,
                  m.g.types.c_str(), m.g.dim, m.g.n_exp, m.g.k_exp, m.g.lat, m.o.max_ctx);
    v.emplace_back(b);
    std::snprintf(b, sizeof b, "nh: experts in VRAM %ld / %ld (%.1f GB), pinned %.1f GB, staging %d slots, CPU lane %d "
                  "threads, load %.1f s", m.st.resident, m.st.experts, m.st.vram_gb, m.st.pinned_gb, m.n_slots,
                  m.st.cpu_threads, m.st.load_s);
    v.emplace_back(b);
    return v;
}
bool Model::save_profile(const std::string& path, std::string& err) const {
    std::ofstream f(path, std::ios::binary);
    if (!f.write((const char*) p->prof.data(), (std::streamsize) (p->prof.size() * 4))) {
        err = path + ": cannot write";
        return false;
    }
    return true;
}

}  // namespace nh
