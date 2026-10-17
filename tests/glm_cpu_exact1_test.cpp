// Real Maya GGUF experts: raw dots, clamped split-base output, quantized intermediate,
// down projection, partial row ranges and multi-token fallback against this build's ggml.
// Run on the rig, with STRATA_GLM_CPU_EXACT1=0/1 and DIAG=1. Dump identities must match.
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace cpu = strata::kernels::cpu;
namespace fs = std::filesystem;
namespace {
uint64_t values = 0, bytes = 0, gate_clamps = 0, up_clamps = 0, cases = 0;
void need(bool ok, const char* what) { if (!ok) throw std::runtime_error(what); }
void compare(const std::vector<float>& a, const std::vector<float>& b, const char* what) {
    need(a.size() == b.size(), "length");
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::memcmp(&a[i], &b[i], 4) || !std::isfinite(a[i])) {
            std::fprintf(stderr, "mismatch %s row=%zu want=%.9g got=%.9g\n", what, i, a[i], b[i]);
            throw std::runtime_error(what);
        }
    }
    values += a.size();
}
std::vector<uint8_t> read(const fs::path& p, uint64_t offset, size_t n) {
    std::ifstream f(p, std::ios::binary); need(f.good(), "open shard");
    f.seekg((std::streamoff) offset);
    std::vector<uint8_t> v(n);
    f.read((char*) v.data(), (std::streamsize) n); need(f.good(), "read shard");
    bytes += n;
    return v;
}
void write(std::ofstream& dump, const void* p, size_t n) {
    dump.write((const char*) p, (std::streamsize) n); need(dump.good(), "write dump");
}
}

int main(int argc, char** argv) {
    try {
        need(argc == 4, "usage: test <pack> <output.bin> <expected-active:0|1>");
        need(cpu::cpu_avx2_ok() && !cpu::cpu_avx512_ok(), "rig requires non-AVX512 AVX2");
        const fs::path pack(argv[1]);
        need(!fs::exists(argv[2]), "preserve existing dump");
        std::ofstream dump(argv[2], std::ios::binary); need(dump.good(), "dump open");
        std::ifstream index(pack / "native_experts.txt"); need(index.good(), "index open");
        std::string line, default_shard;
        int layers = 0;
        for (; std::getline(index, line);) {
            if (line.empty()) continue;
            if (line[0] == '#') {
                const std::string key = "absolute offsets in ";
                const auto a = line.find(key);
                if (a != std::string::npos) {
                    const auto start = a + key.size(), end = line.find(',', start);
                    default_shard = line.substr(start, end - start);
                }
                continue;
            }
            std::istringstream ss(line);
            int layer, gu, dt;
            uint64_t offset, blob_bytes, go, uo, dno;
            need(bool(ss >> layer >> gu >> dt >> offset >> blob_bytes >> go >> uo >> dno), "index row");
            std::vector<std::string> shards;
            for (std::string w; ss >> w;) shards.push_back(w);
            if (shards.empty()) shards.assign(3, default_shard);
            if (shards.size() == 1) {
                const std::string shard = shards[0];
                shards.assign(3, shard);   // keep the value alive if assign reallocates
            }
            need(shards.size() == 3 && gu == 16 && (dt == 18 || dt == 22), "unexpected pack format");
            cpu::NativeFmt f; std::string err;
            need(cpu::native_fmt(gu, dt, 4096, 2048, f, err), "native fmt");
            need(f.bytes == blob_bytes, "geometry drift");
            const int H = (int) f.n_embd, FF = (int) f.n_ff;
            const auto dot = ggml_get_type_traits_cpu((ggml_type) gu)->vec_dot;
            const auto down = ggml_get_type_traits_cpu((ggml_type) dt)->vec_dot;
            for (int e : {0, 37, 143, 287}) {
                // Separate host allocations, as well as non-adjacent source tensors.
                auto gate = read(pack.parent_path() / shards[0], go + e * f.up_off, f.up_off);
                auto up = read(pack.parent_path() / shards[1], uo + e * f.up_off, f.up_off);
                const size_t db = (size_t) H * f.d_row;
                auto dn = read(pack.parent_path() / shards[2], dno + e * db, db);
                for (int seed = 0; seed < 4; ++seed) {
                    std::mt19937 rng(734621 + 991 * layer + e * 17 + seed);
                    std::vector<float> x(H), gr(FF), ur(FF), gg(FF), ug(FF);
                    const float scale = seed == 0 ? 0.f : seed == 1 ? 0.125f : seed == 2 ? 1.f : 128.f;
                    for (float& v : x) v = ((int) (rng() % 65537) - 32768) * (scale / 32768.f);
                    std::vector<uint8_t> act(f.act_bytes), h_ref(f.h_bytes), h_got(f.h_bytes);
                    cpu::native_quant_act(f, x.data(), act.data());
                    const void* a[2] = {act.data(), act.data()};
                    for (int r = 0; r < FF; ++r) {
                        dot(H, &gr[r], 0, gate.data() + r * f.gu_row, 0, a[0], 0, 1);
                        dot(H, &ur[r], 0, up.data() + r * f.gu_row, 0, a[0], 0, 1);
                        gate_clamps += gr[r] > 10.f;
                        up_clamps += ur[r] < -10.f || ur[r] > 10.f;
                    }
                    cpu::iq2xxs256_rows_exact_one(gate.data(), f.gu_row, H, a[0], gg.data(), 0, FF);
                    cpu::iq2xxs256_rows_exact_one(up.data(), f.gu_row, H, a[0], ug.data(), 0, FF);
                    compare(gr, gg, "raw gate"); compare(ur, ug, "raw up");
                    write(dump, gg.data(), gg.size() * 4); write(dump, ug.data(), ug.size() * 4);
                    for (float limit : {10.f, std::numeric_limits<float>::infinity()}) {
                        std::vector<float> want(FF), got(FF, -12345.f);
                        float* o[1] = {got.data()};
                        for (int r = 0; r < FF; ++r) {
                            float g = gr[r], u = ur[r];
                            if (std::isfinite(limit)) {
                                g = std::fmin(g, limit); u = std::fmin(std::fmax(u, -limit), limit);
                            }
                            want[r] = (g / (1.f + std::exp(-g))) * u;
                        }
                        // Leave sentinels around a nonzero subrange, then cover all rows on six workers.
                        cpu::native_gu_rows_split(f, gate.data(), up.data(), a, 1, o, 17, 31, limit);
                        for (int r = 0; r < FF; ++r) {
                            const float v = r >= 17 && r < 31 ? want[r] : -12345.f;
                            need(!std::memcmp(&got[r], &v, 4), "partial range/sentinel");
                        }
                        values += FF;
                        std::vector<std::thread> workers;
                        for (int t = 0; t < 6; ++t)
                            workers.emplace_back([&, t] {
                                cpu::native_gu_rows_split(f, gate.data(), up.data(), a, 1, o,
                                                          FF * t / 6, FF * (t + 1) / 6, limit);
                            });
                        for (auto& t : workers) t.join();
                        compare(want, got, "split/clamp");
                        write(dump, got.data(), got.size() * 4);
                        cpu::native_quant_h(f, want.data(), h_ref.data());
                        cpu::native_quant_h(f, got.data(), h_got.data());
                        need(h_ref == h_got, "quantized intermediate");
                        write(dump, h_got.data(), h_got.size());
                        std::vector<float> dr(H), dg(H);
                        for (int r = 0; r < H; ++r) down(FF, &dr[r], 0, dn.data() + r * f.d_row, 0, h_ref.data(), 0, 1);
                        const void* h[1] = {h_got.data()}; float* out[1] = {dg.data()};
                        cpu::native_down_rows_split(f, dn.data(), h, 1, out, 0, H);
                        compare(dr, dg, "down"); write(dump, dg.data(), dg.size() * 4);
                        if (limit == 10.f) {
                            // #1544 guard: nt=2 must still use the old split-base ggml arithmetic.
                            const auto before = cpu::native_split_counts();
                            std::vector<float> m0(FF), m1(FF); float* multi[2] = {m0.data(), m1.data()};
                            cpu::native_gu_rows_split(f, gate.data(), up.data(), a, 2, multi, 0, FF, limit);
                            need(cpu::native_split_counts().exact_rows == before.exact_rows, "multi-token dispatch changed");
                            compare(want, m0, "nt2 first"); compare(want, m1, "nt2 second");
                        }
                    }
                    ++cases;
                }
            }
            ++layers;
            std::printf("EXACT1_LAYER_PASS layer=%d gu=%d down=%d cases=%llu\n", layer, gu, dt, (unsigned long long) cases);
            std::fflush(stdout);
        }
        const auto count = cpu::native_split_counts();
        const bool active = std::atoi(argv[3]) == 1;
        need(layers == 42 && cases == 672 && gate_clamps > 0 && up_clamps > 0, "coverage");
        need(count.exact_rows == (active ? cases * 2 * (2048 + 14) : 0), "exact activation");
        need(count.ggml_rows == cases * 4096 + (active ? 0 : cases * 2 * (2048 + 14)), "fallback activation");
        dump.close(); need(dump.good(), "dump close");
        std::printf("EXACT1_NATIVE_PASS layers=%d cases=%llu bitwise_values=%llu expert_bytes=%llu exact_rows=%llu ggml_rows=%llu gate_clamps=%llu up_clamps=%llu\n",
                    layers, (unsigned long long) cases, (unsigned long long) values, (unsigned long long) bytes,
                    (unsigned long long) count.exact_rows, (unsigned long long) count.ggml_rows,
                    (unsigned long long) gate_clamps, (unsigned long long) up_clamps);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "EXACT1_NATIVE_FAIL %s\n", e.what()); return 1;
    }
}
