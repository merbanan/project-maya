// src/core/glm_fast_path.cu - the glm5-next FAST decode path (Glm5Model members).
//
// The same model as Glm5Model::step_layers (the correctness-first reference, STRATA_GLM_SLOW=1), run
// as ~11 fused launches per layer (strata/kernels/glm_fast.hpp) on a private non-blocking stream per
// device, with NO host synchronisation inside a token except where an expert is missing:
//
//   * every weight pointer is resolved once at load (no string lookups per layer per token);
//   * the router runs on the device and looks its experts up in a DEVICE table (layer x expert ->
//     VRAM slot pointer); an all-resident layer never waits for the host;
//   * every route is published to a host-mapped ring; a per-device SERVICE thread keeps the LFU
//     counts and, when a route has misses, reads the missing experts from the shards (pread into
//     pinned staging, in parallel), DMAs them into victim slots OF THAT LAYER on a copy stream and
//     answers through host-mapped memory - the device spins in a one-block wait kernel meanwhile.
//
// Victims come only from the requesting layer's own slot partition, so no kernel can be reading a
// slot while it is overwritten: the device is parked in that layer's wait kernel, the layer's
// resident experts are excluded, and every other layer's slots are untouched.
#include "glm_kv_stream.hpp"
#include "glm_fast_state.hpp"
#include "strata/core/glm_model.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/glm_fast.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include "ggml.h"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#if !defined(STRATA_USE_HIP)
#include <nvtx3/nvToolsExt.h>
#endif

#include <algorithm>
#include <cctype>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <cstring>
#include <fstream>
#include <functional>
#include <sstream>
#include <mutex>
#include <thread>
#include <vector>
#include <bit>
#ifndef _WIN32
#include <unistd.h>
#else
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#ifdef __linux__
#include <sched.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#endif

namespace gf = strata::kernels::glmf;

// Keeps the GPU out of its idle P-state for ns nanoseconds.  A consumer card at P8 drops its PCIe link (Gen1 on an
// RTX 4070 Ti SUPER / 5070 Ti), and copies alone do not wake it, so a link timed at idle reads 3-8x slower than it
// runs during decode (Tesla cards keep the link up, so this never showed there).
static __global__ void glm_link_wake(unsigned long long ns) {
#if defined(STRATA_USE_HIP)
    // no PTX %globaltimer on AMD: wall_clock64() is the constant 100 MHz counter (10 ns a tick) on gfx11 / gfx12, and
    // a wake-up spin needs no exact timing (from @boxwrench, #24)
    const long long t0 = wall_clock64();
    const long long ticks = (long long) (ns / 10ull);
    while (wall_clock64() - t0 < ticks) {
    }
#else
    unsigned long long t0, t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    do {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    } while (t - t0 < ns);
#endif
}

namespace strata::core {

using glmfast::cpu_relax;
using glmfast::Workers;

// The NUMA nodes this machine has (sysfs "online" list), and the most of this process's CPUs on any one of them;
// {} / 0 when unknown or on a single node.
static std::vector<int> numa_nodes() {
    std::vector<int> nodes;
#ifdef __linux__
    std::string list;
    std::ifstream("/sys/devices/system/node/online") >> list;
    std::stringstream ss(list);
    std::string r;
    while (std::getline(ss, r, ',')) {
        const size_t d = r.find('-');
        const int a = std::atoi(r.c_str()), b = d == std::string::npos ? a : std::atoi(r.c_str() + d + 1);
        for (int n = a; n <= b && n < 256; ++n) nodes.push_back(n);
    }
#endif
    return nodes;
}

// this process's CPUs on each NUMA node (sysfs cpulist), node by node; {} when unknown or on a single node
static std::vector<std::vector<int>> numa_node_cpu_lists() {
    std::vector<std::vector<int>> out;
#ifdef __linux__
    const auto nodes = numa_nodes();
    if (nodes.size() < 2) return out;
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof set, &set) != 0) return out;
    for (int n : nodes) {
        std::string list;
        std::ifstream("/sys/devices/system/node/node" + std::to_string(n) + "/cpulist") >> list;
        std::stringstream ss(list);
        std::string r;
        std::vector<int> cpus;
        while (std::getline(ss, r, ',')) {
            const size_t d = r.find('-');
            const int a = std::atoi(r.c_str()), b = d == std::string::npos ? a : std::atoi(r.c_str() + d + 1);
            for (int c = a; c <= b && c < CPU_SETSIZE; ++c)
                if (CPU_ISSET(c, &set)) cpus.push_back(c);
        }
        if (!cpus.empty()) out.push_back(std::move(cpus));
    }
#endif
    return out;
}

// this process's CPUs (its affinity mask) by NUMA node: {node, CPUs}, node by node - one group (node -1) without NUMA
static std::vector<std::pair<int, std::vector<int>>> cpu_groups() {
    std::vector<std::pair<int, std::vector<int>>> out;
#ifdef __linux__
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof set, &set) != 0) return out;
    for (int n : numa_nodes()) {
        std::string list;
        std::ifstream("/sys/devices/system/node/node" + std::to_string(n) + "/cpulist") >> list;
        std::stringstream ss(list);
        std::string r;
        std::vector<int> cpus;
        while (std::getline(ss, r, ',')) {
            const size_t d = r.find('-');
            const int a = std::atoi(r.c_str()), b = d == std::string::npos ? a : std::atoi(r.c_str() + d + 1);
            for (int c = a; c <= b && c < CPU_SETSIZE; ++c)
                if (CPU_ISSET(c, &set)) cpus.push_back(c);
        }
        if (!cpus.empty()) out.push_back({n, std::move(cpus)});
    }
    if (out.empty()) {
        std::vector<int> cpus;
        for (int c = 0; c < CPU_SETSIZE; ++c)
            if (CPU_ISSET(c, &set)) cpus.push_back(c);
        if (!cpus.empty()) out.push_back({-1, std::move(cpus)});
    }
#endif
    return out;
}

// a CPU's physical core (package, core id): its SMT siblings share it
static std::pair<int, int> cpu_core(int cpu) {
    int pkg = cpu, core = cpu;
#ifdef __linux__
    const std::string t = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
    std::ifstream(t + "physical_package_id") >> pkg;
    std::ifstream(t + "core_id") >> core;
#endif
    return {pkg, core};
}

// the NUMA node a GPU hangs off (sysfs, by its PCI address), -1 when unknown (a VM, or no NUMA)
static int gpu_numa_node(int dev) {
    int node = -1;
#ifdef __linux__
    char bus[32] = {0};
    if (cudaDeviceGetPCIBusId(bus, (int) sizeof bus, dev) != cudaSuccess) {
        cudaGetLastError();
        return -1;
    }
    std::string b(bus);
    for (char& ch : b) ch = (char) std::tolower((unsigned char) ch);
    std::ifstream("/sys/bus/pci/devices/" + b + "/numa_node") >> node;
#else
    (void) dev;
#endif
    return node;
}

// Part `part` of a split across `n` GPUs (`gpu_node`: each part's GPU's NUMA node): the CPUs its pool runs on - no
// other part's, on any machine - with the node it is on (-1: none known) and how many of them to leave free.  Whole
// NUMA nodes while there are as many as parts: each part on its GPU's node when the GPUs hang off distinct nodes,
// else part i on node i.  With fewer nodes than parts (one socket), the parts on a node share it out in runs of whole
// physical cores (SMT siblings together).  The 4 CPUs a node keeps free (the main thread's event wait, the service
// and warm-up threads, as one GPU's pool leaves) are shared by its parts.  {} with nothing to give.
static std::vector<int> part_cpus(int part, int n, const std::vector<int>& gpu_node, int& node, int& spare) {
    const auto groups = cpu_groups();
    node = -1;
    spare = 4;
    if (groups.empty() || n < 1 || part < 0 || part >= n) return {};
    const int G = (int) groups.size();
    std::vector<int> grp((size_t) n);
    bool by_gpu = n <= G && (int) gpu_node.size() == n;
    std::vector<bool> taken((size_t) G, false);
    for (int i = 0; by_gpu && i < n; ++i) {
        int at = -1;
        for (int j = 0; j < G; ++j)
            if (groups[(size_t) j].first >= 0 && groups[(size_t) j].first == gpu_node[(size_t) i]) at = j;
        by_gpu = at >= 0 && !taken[(size_t) at];
        if (by_gpu) {
            taken[(size_t) at] = true;
            grp[(size_t) i] = at;
        }
    }
    if (!by_gpu)
        for (int i = 0; i < n; ++i) grp[(size_t) i] = i % G;
    int k = 0, slot = 0;   // the parts on this part's node, and this part's place among them
    for (int i = 0; i < n; ++i)
        if (grp[(size_t) i] == grp[(size_t) part]) {
            if (i == part) slot = k;
            ++k;
        }
    const auto& g = groups[(size_t) grp[(size_t) part]];
    node = g.first;
    spare = (4 + k - 1) / k;
    if (k == 1) return g.second;
    std::vector<std::pair<std::pair<int, int>, int>> by_core;
    for (int c : g.second) by_core.push_back({cpu_core(c), c});
    std::sort(by_core.begin(), by_core.end());
    std::vector<std::vector<int>> cores;
    for (size_t i = 0; i < by_core.size(); ++i) {
        if (i == 0 || by_core[i].first != by_core[i - 1].first) cores.emplace_back();
        cores.back().push_back(by_core[i].second);
    }
    std::vector<int> out;
    const size_t a = cores.size() * (size_t) slot / (size_t) k, b = cores.size() * (size_t) (slot + 1) / (size_t) k;
    for (size_t j = a; j < b; ++j) out.insert(out.end(), cores[j].begin(), cores[j].end());
    std::sort(out.begin(), out.end());
    return out;
}

static int numa_node_cpus() {
    int best = 0;
    for (const auto& c : numa_node_cpu_lists()) best = std::max(best, (int) c.size());
    return best;
}

// The RAM tier's pinned memory spread page by page over every NUMA node: mmap + mbind(MPOL_INTERLEAVE) (+ transparent
// huge pages), then cudaHostRegister - the pages are placed when the registration faults them in, so the policy
// holds.  cudaHostAlloc's pages come from the driver node by node whatever the thread's policy (a 112 GB tier: 57.6 GB
// on one node, 72.7 GB on the other, each expert on ONE of them), and the CPU lane reads every expert with all its
// threads through that node's memory controllers only (2-socket Xeon, Q3_K experts, 44 threads: 216 us an expert
// node-local, 152 us interleaved).  Mapped, so the device reads its host pointers as it does cudaHostAlloc's.
// nullptr: one node, STRATA_GLM_NUMA=0, no host-pointer access for registered memory, or a failure.
static void* numa_pinned(size_t bytes) {
#ifdef __linux__
    const char* v = getenv("STRATA_GLM_NUMA");
    const auto nodes = numa_nodes();
    if ((v != nullptr && std::atoi(v) == 0) || nodes.size() < 2) return nullptr;
    int dev = 0, ok = 0;
    cudaGetDevice(&dev);
    if (cudaDeviceGetAttribute(&ok, cudaDevAttrCanUseHostPointerForRegisteredMem, dev) != cudaSuccess || !ok) {
        cudaGetLastError();
        return nullptr;
    }
    void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (p == MAP_FAILED) return nullptr;
    unsigned long mask[4] = {0, 0, 0, 0};
    for (int n : nodes) mask[n / 64] |= 1ul << (n % 64);
    // huge pages only while free 2 MB blocks cover the tier (/proc/buddyinfo, orders 9 and up): otherwise every 2 MB
    // fault of a MADV_HUGEPAGE region compacts memory on the spot (transparent_hugepage/defrag "madvise") and mostly
    // fails - a split's second card, pinning its tier after the first card's had taken most of the RAM and the page
    // cache the rest (0.28 GB left in 2 MB blocks), sat in the kernel for over 10 minutes at ~10 MB/s.  4 KB pages
    // reclaim the cache as they fault in (the CPU lane reads them 6-8% slower than huge ones)
    size_t huge_free = 0;
    bool huge_known = false;
    if (FILE* bf = std::fopen("/proc/buddyinfo", "r")) {
        char line[512];
        while (std::fgets(line, sizeof line, bf)) {
            const char* z = std::strstr(line, "zone");
            if (z == nullptr) continue;
            z += 4;
            while (*z == ' ') ++z;
            while (*z != ' ' && *z != '\0') ++z;   // the zone's name
            unsigned long long c[16] = {};
            int k = 0;
            char* end = nullptr;
            for (const char* q = z; k < 16; ++k, q = end) {
                c[k] = std::strtoull(q, &end, 10);
                if (end == q) break;
            }
            for (int o = 9; o < k; ++o) huge_free += (size_t) c[o] << (o + 12);
            huge_known = true;
        }
        std::fclose(bf);
    }
    if (!huge_known || huge_free >= bytes + ((size_t) 1 << 30)) madvise(p, bytes, MADV_HUGEPAGE);
    // (the registration faults the pages in, one thread: faulting them from 16 threads first started ~40 s sooner but
    // left fewer huge pages - the CPU lane read its experts 6-8% slower, 2-socket Xeon 6152)
    // fault in before registering: cudaHostRegister holds the driver lock, stalling other parts' copies
    if (syscall(SYS_mbind, p, bytes, 3 /* MPOL_INTERLEAVE */, mask, (unsigned long) (sizeof mask * 8), 0u) != 0) {
        munmap(p, bytes);
        return nullptr;
    }
    for (size_t o = 0; o < bytes; o += 4096) ((volatile char*) p)[o] = 0;
    if (cudaHostRegister(p, bytes, cudaHostRegisterPortable | cudaHostRegisterMapped) != cudaSuccess) {
        cudaGetLastError();
        munmap(p, bytes);
        return nullptr;
    }
    return p;
#else
    (void) bytes;
    return nullptr;
#endif
}

int64_t glmfast::zfs_arc_reclaimable() {
#ifdef __linux__
    FILE* f = std::fopen("/proc/spl/kstat/zfs/arcstats", "r");   // (rows "name type value" after two header lines)
    if (f == nullptr) return 0;
    long long size = -1, cmin = -1;
    char line[256], name[64];
    while (std::fgets(line, sizeof line, f)) {
        int type = 0;
        long long v = 0;
        if (std::sscanf(line, "%63s %d %lld", name, &type, &v) != 3) continue;
        if (std::strcmp(name, "size") == 0) size = v;
        else if (std::strcmp(name, "c_min") == 0) cmin = v;
    }
    std::fclose(f);
    return size > 0 && cmin >= 0 && size > cmin ? (int64_t) (size - cmin) : 0;
#else
    return 0;
#endif
}

// the RAM free now: MemAvailable and ZFS's reclaimable ARC (Windows: the smaller of the free RAM and the free commit,
// which pinning is charged to); 0 when unknown
static int64_t avail_ram_now() {
#ifdef _WIN32
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof ms;
    return GlobalMemoryStatusEx(&ms) ? (int64_t) std::min(ms.ullAvailPhys, ms.ullAvailPageFile) : 0;
#else
    long long kb = 0;
    if (FILE* f = std::fopen("/proc/meminfo", "r")) {
        char line[256];
        while (std::fgets(line, sizeof line, f))
            if (std::sscanf(line, "MemAvailable: %lld kB", &kb) == 1) break;
        std::fclose(f);
    }
    return kb > 0 ? (int64_t) kb * 1024 + glmfast::zfs_arc_reclaimable() : 0;
#endif
}

static void numa_pinned_free(void* p, size_t bytes) {
#ifdef __linux__
    cudaHostUnregister(p);
    munmap(p, bytes);
#else
    (void) p;
    (void) bytes;
#endif
}

namespace glmfast {

// [off, off + len) of a shard into dst: O_DIRECT through an aligned per-thread bounce buffer when the shard has a
// direct fd (no page cache), a buffered pread otherwise, the mapping as the last resort
void read_slice(const Glm5Model::Shard& sh, uint64_t off, size_t len, uint8_t* dst) {
#ifndef _WIN32
    if (sh.fd_direct >= 0) {
        // freed when its thread ends
        struct Bounce {
            uint8_t* p = nullptr;
            size_t cap = 0;
            ~Bounce() { std::free(p); }
        };
        static thread_local Bounce bb;
        uint8_t*& bounce = bb.p;
        size_t& cap = bb.cap;
        const uint64_t a0 = off & ~(uint64_t) 4095, a1 = (off + len + 4095) & ~(uint64_t) 4095;
        const size_t need = (size_t) (a1 - a0);
        if (cap < need) {
            std::free(bounce);
            bounce = (uint8_t*) std::aligned_alloc(4096, need);
            cap = bounce ? need : 0;
        }
        if (bounce != nullptr) {
            size_t got = 0;
            while (got < need) {
                const ssize_t r = pread(sh.fd_direct, bounce + got, need - got, (off_t) (a0 + got));
                if (r <= 0) break;   // EOF: the last aligned read is short where the file ends
                got += (size_t) r;
            }
            if (got >= (size_t) (off - a0) + len) {
                std::memcpy(dst, bounce + (off - a0), len);
                return;
            }
        }
    }
    size_t done = 0;
    while (done < len) {
        const ssize_t r = pread(sh.fd, dst + done, len - done, (off_t) (off + done));
        if (r <= 0) {
            std::memcpy(dst + done, sh.base + off + done, len - done);
            return;
        }
        done += (size_t) r;
    }
#else
    // Windows: the unbuffered handle the same way (sector-aligned offset, size and buffer), the mapping otherwise.
    // The handle is overlapped (reads from many threads run at once): each read waits on this thread's own event
    if (sh.h_direct != nullptr) {
        // freed and closed when its thread ends
        struct Bounce {
            uint8_t* p = nullptr;
            size_t cap = 0;
            HANDLE ev = CreateEventA(nullptr, TRUE, FALSE, nullptr);
            ~Bounce() {
                _aligned_free(p);
                if (ev != nullptr) CloseHandle(ev);
            }
        };
        static thread_local Bounce bb;
        uint8_t*& bounce = bb.p;
        size_t& cap = bb.cap;
        const HANDLE done_ev = bb.ev;
        const uint64_t a0 = off & ~(uint64_t) 4095, a1 = (off + len + 4095) & ~(uint64_t) 4095;
        const size_t need = (size_t) (a1 - a0);
        if (cap < need) {
            _aligned_free(bounce);
            bounce = (uint8_t*) _aligned_malloc(need, 4096);
            cap = bounce ? need : 0;
        }
        if (bounce != nullptr && done_ev != nullptr) {
            size_t got = 0;
            while (got < need) {
                OVERLAPPED ov{};
                const uint64_t at = a0 + got;
                ov.Offset = (DWORD) at;
                ov.OffsetHigh = (DWORD) (at >> 32);
                ov.hEvent = done_ev;
                const DWORD want = (DWORD) std::min<size_t>(need - got, (size_t) 1 << 30);
                DWORD r = 0;
                if (!ReadFile((HANDLE) sh.h_direct, bounce + got, want, nullptr, &ov) &&
                    GetLastError() != ERROR_IO_PENDING) break;   // EOF: the last aligned read is short
                if (!GetOverlappedResult((HANDLE) sh.h_direct, &ov, &r, TRUE) || r == 0) break;
                got += (size_t) r;
            }
            if (got >= (size_t) (off - a0) + len) {
                std::memcpy(dst, bounce + (off - a0), len);
                return;
            }
        }
    }
    if (sh.base == nullptr) {   // the views went after the load (pack_release_views): nothing to fall back to
        std::fprintf(stderr, "glm fast: an expert read of %s failed (error %lu) - STRATA_GLM_KEEP_MAP=1 keeps the "
                             "mapping to fall back to\n", sh.path.c_str(), (unsigned long) GetLastError());
        std::abort();
    }
    std::memcpy(dst, sh.base + off, len);
#endif
}

}  // namespace glmfast

using glmfast::read_slice;

// the residual from a token's embedding row: GLM's 4 mHC streams, hy_v3's single stream
static void embed_in(const Glm5Geometry& g, const float* emb, float* R, cudaStream_t s) {
    if (g.gqa()) cudaMemcpyAsync(R, emb, (size_t) g.n_embd * sizeof(float), cudaMemcpyDeviceToDevice, s);
    else gf::embed_streams(emb, R, g.n_embd, s);
}
// the head's input: output_norm of the residual (GLM: of its 4 streams' mean) as f32 and q8_1
static void head_in(const Glm5Geometry& g, float* R, const float* norm_w, float* x, void* xq, cudaStream_t s) {
    if (g.gqa()) gf::rms_q8(R, nullptr, norm_w, g.norm_eps, g.n_embd, x, xq, s);
    else gf::head_prep(R, norm_w, g.norm_eps, g.n_embd, x, xq, s);
}

static float* fa_take(uint8_t*& at, size_t floats) {
    float* p = (float*) at;
    at += (floats * sizeof(float) + 255u) & ~(size_t) 255u;
    return p;
}
static void* fa_take_b(uint8_t*& at, size_t bytes) {
    void* p = at;
    at += (bytes + 255u) & ~(size_t) 255u;
    return p;
}

bool Glm5Model::fast_setup(std::string& err) {
    cudaSetDevice(dev_);
    auto* F = new FastState();
    fast_ = F;
    // Not on Windows: there an APU's GPU memory is a fixed carve-out (AMD's Variable Graphics Memory, or the BIOS)
    // that Windows does not count as RAM, so it is sized like a discrete card - the pool from what HIP reports free,
    // the RAM tier from the free RAM and commit (below)
#if defined(STRATA_USE_HIP) && !defined(_WIN32)
    int integrated = 0;
    if (cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, dev_) == cudaSuccess)
        F->unified_memory = integrated != 0;
    else
        cudaGetLastError();
#endif
    const Glm5Geometry& g = g_;
    F->timing = getenv("STRATA_GLM_TIMING") != nullptr;
    F->prof_on = getenv("STRATA_GLM_PROF") != nullptr;
    if (F->prof_on) F->pskip = (uint64_t) std::max(0, std::atoi(getenv("STRATA_GLM_PROF")));
    if (const char* pn = getenv("STRATA_GLM_PREFETCH_N")) {
        F->max_pf = std::max(0, std::min(gf::kSpares - 1, std::atoi(pn)));
        // said at start: the prefetch's settings are read once and nothing else shows them (#6)
        if (F->max_pf > 0) {
            const char* at = getenv("STRATA_GLM_PREFETCH_AT");
            const char* bl = getenv("STRATA_GLM_PREFETCH_BLOCKS");
            const char* rk = getenv("STRATA_GLM_PREFETCH_RANK");
            std::fprintf(stderr, "glm fast: CUDA%d prefetch on: up to %d of the next layer's predicted experts a layer%s, "
                                 "the copy at %s, %s blocks, %s\n", dev_, F->max_pf,
                         std::atoi(pn) > F->max_pf ? " (STRATA_GLM_PREFETCH_N is at most 2)" : "",
                         at != nullptr && (std::strcmp(at, "fetch") == 0 || std::strcmp(at, "cpu") == 0) ? at : "route",
                         bl != nullptr ? bl : "half the SMs'",
                         rk != nullptr ? (std::string("prediction ranks 1-") + rk).c_str() : "any prediction rank");
        }
    }
    // RAM-resident mode (--glm-ram-resident / STRATA_GLM_RAM_RESIDENT=1): the RAM tier's per-class cap grows by
    // ram_slack slots per MoE layer so it holds EVERY expert VRAM does not; nothing is ever dropped to disk and the
    // start fails when the tier cannot hold them all - the disk is the pack's home, never a read source at runtime
    F->ram_resident = getenv("STRATA_GLM_RAM_RESIDENT") != nullptr && std::atoi(getenv("STRATA_GLM_RAM_RESIDENT")) != 0;
    if (const char* rs = getenv("STRATA_GLM_RAM_SLACK")) F->ram_slack = std::max(0, std::atoi(rs));
    else if (F->ram_resident) F->ram_slack = 16;
    if (cudaStreamCreateWithFlags(&F->cs, cudaStreamNonBlocking) != cudaSuccess ||
        cudaStreamCreateWithFlags(&F->copy, cudaStreamNonBlocking) != cudaSuccess ||
        cudaStreamCreateWithFlags(&F->ps, cudaStreamNonBlocking) != cudaSuccess ||
        cudaEventCreateWithFlags(&F->ev_hop, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&F->ev_pred, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&F->ev_pf, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&F->ev_pf_prev, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&F->ev_done, cudaEventDisableTiming) != cudaSuccess) {
        err = "glm fast: streams/events did not create";
        return false;
    }

    // ---- the weights, resolved once
    const int NL = g.n_layers + 1;   // every per-layer table: the trunk + the NextN block's slot
    F->L.assign((size_t) NL, FastState::Layer{});
    std::string missing;
    for (int il = l0_; il < lt_; ++il) {
        const bool is_mtp = il == mtp_il_;
        auto& Ly = F->L[(size_t) il];
        const std::string P = "blk." + std::to_string(il) + ".";
        const auto f32 = [&](const char* n) -> const float* {
            auto it = w_.find(P + n);
            if (it == w_.end()) { missing += P + n + " (f32) "; return nullptr; }
            return it->second;
        };
        const auto b16 = [&](const char* n) -> const uint16_t* {
            auto it = w16_.find(P + n);
            if (it == w16_.end()) { missing += P + n + " (bf16) "; return nullptr; }
            return it->second;
        };
        const auto q = [&](const char* n) -> WSlot {
            auto it = ws_map_.find(P + n);
            if (it == ws_map_.end() || it->second.type == 0 || !gf::mv_supported(it->second.type)) {
                missing += P + n + " (quant) ";
                return WSlot{};
            }
            return it->second;
        };
        // a projection a GGUF keeps unquantized (BF16 - e.g. attn_kv_a_mqa in some community GGUFs) where Maya's
        // quants quantize it: the BF16 copy; the jobs that take these pass the float activation as well
        const auto q_or_b16 = [&](const char* n, int64_t n_in, int64_t n_out) -> WSlot {
            const auto it = ws_map_.find(P + n);
            if (it == ws_map_.end() || it->second.type == 0) {
                const auto b = w16_.find(P + n);
                if (b != w16_.end()) return WSlot{nullptr, gf::kTypeBF16, b->second, n_in, n_out};
            }
            return q(n);
        };
        Ly.recr = is_mtp ? false : g.is_recr(il);
        Ly.moe = il >= g.dense_lead;
        Ly.mtp = is_mtp;
        Ly.gqa = g.gqa();
        Ly.dsv4 = g.dsv4() && !is_mtp;
        if (Ly.gqa) {
            // hy_v3: plain pre-norm residuals, grouped-query attention, the router in F32; the draft block is a Hy3
            // layer read through eh_proj, its output through the shared head's norm
            if (is_mtp) {
                Ly.eh = q("nextn.eh_proj.weight");
                Ly.enorm = f32("nextn.enorm.weight");
                Ly.hnorm = f32("nextn.hnorm.weight");
                Ly.shnorm = f32("nextn.shared_head_norm.weight");
            }
            Ly.attn_norm = f32("attn_norm.weight");
            Ly.ffn_norm = f32("ffn_norm.weight");
            Ly.q = q("attn_q.weight");
            Ly.k = q("attn_k.weight");
            Ly.v = q("attn_v.weight");
            Ly.out = q("attn_output.weight");
            Ly.q_norm = f32("attn_q_norm.weight");
            Ly.k_norm = f32("attn_k_norm.weight");
        } else if (is_mtp) {
            // no hyper-connections: eh_proj in, plain residuals, the shared head's norm out
            Ly.eh = q("nextn.eh_proj.weight");
            Ly.enorm = f32("nextn.enorm.weight");
            Ly.hnorm = f32("nextn.hnorm.weight");
            Ly.shnorm = f32("nextn.shared_head_norm.weight");
        } else {
            Ly.hc_attn_fn = b16("hc_attn_fn.weight");
            Ly.hc_ffn_fn = b16("hc_ffn_fn.weight");
            Ly.hc_attn_scale = f32("hc_attn_scale.weight");
            Ly.hc_attn_base = f32("hc_attn_base.weight");
            Ly.hc_ffn_scale = f32("hc_ffn_scale.weight");
            Ly.hc_ffn_base = f32("hc_ffn_base.weight");
        }
        if (!Ly.gqa) {
        Ly.attn_norm = f32("attn_norm.weight");
        Ly.ffn_norm = f32("ffn_norm.weight");
        if (Ly.recr) {
            Ly.q = q("attn_q.weight");
            Ly.k = q("attn_k.weight");
            Ly.v = q("attn_v.weight");
            Ly.f_a = b16("ssm_f_a.weight");
            Ly.g_a = b16("ssm_g_a.weight");
            Ly.f_b = b16("ssm_f_b.weight");
            Ly.g_b = b16("ssm_g_b.weight");
            Ly.beta = b16("ssm_beta.weight");
            Ly.conv[0] = f32("ssm_conv1d_q.weight");
            Ly.conv[1] = f32("ssm_conv1d_k.weight");
            Ly.conv[2] = f32("ssm_conv1d_v.weight");
            Ly.dt_bias = f32("ssm_dt.bias");
            Ly.ssm_a = f32("ssm_a");
            Ly.ssm_norm = f32("ssm_norm.weight");
        } else if (Ly.dsv4) {
            // deepseek4: the q LoRA, one K/V head, the sinks, the grouped output; the compressors (ratio > 0) and the
            // indexer (ratio 4) in FP16
            const int r = g.ratio(il);
            Ly.q_a = q("attn_q_a.weight");
            Ly.q_b = q("attn_q_b.weight");
            Ly.kv_a = q("attn_kv.weight");
            Ly.q_a_norm = f32("attn_q_a_norm.weight");
            Ly.kv_a_norm = f32("attn_kv_a_norm.weight");
            Ly.sinks = f32("attn_sinks.weight");
            Ly.o_a = q("attn_output_a.weight");
            Ly.o_b = q("attn_output_b.weight");
            if (r > 0) {
                Ly.c_kv = b16("attn_compressor_kv.weight");
                Ly.c_gate = b16("attn_compressor_gate.weight");
                Ly.c_ape = f32("attn_compressor_ape.weight");
                Ly.c_norm = f32("attn_compressor_norm.weight");
            }
            if (r == 4) {
                Ly.ic_kv = b16("indexer_compressor_kv.weight");
                Ly.ic_gate = b16("indexer_compressor_gate.weight");
                Ly.ic_ape = f32("indexer_compressor_ape.weight");
                Ly.ic_norm = f32("indexer_compressor_norm.weight");
                Ly.idx_q_b = b16("indexer.attn_q_b.weight");
                Ly.idx_proj = b16("indexer.proj.weight");
            }
            for (const char* n : {"attn_compressor_kv.weight", "attn_compressor_gate.weight",
                                  "indexer_compressor_kv.weight", "indexer_compressor_gate.weight",
                                  "indexer.attn_q_b.weight", "indexer.proj.weight", "hc_attn_fn.weight", "hc_ffn_fn.weight"})
                if (w16_.count(P + n) != 0 && w16h_.count(P + n) == 0) missing += P + n + " (fp16 expected) ";
        } else {
            Ly.q_a = q_or_b16("attn_q_a.weight", g.n_embd, g.q_lora);
            Ly.q_b = q("attn_q_b.weight");
            Ly.kv_a = q_or_b16("attn_kv_a_mqa.weight", g.n_embd, g.kv_lora);
            Ly.q_a_norm = f32("attn_q_a_norm.weight");
            Ly.kv_a_norm = f32("attn_kv_a_norm.weight");
            Ly.k_norm_w = f32("indexer.k_norm.weight");
            Ly.k_norm_b = f32("indexer.k_norm.bias");
            Ly.ape = f32("indexer_compressor_ape.weight");
            Ly.idx_k = b16("indexer.attn_k.weight");
            Ly.idx_gate = b16("indexer_compressor_gate.weight");
            Ly.idx_q_b = b16("indexer.attn_q_b.weight");
            Ly.idx_proj = b16("indexer.proj.weight");
            Ly.k_b = b16("attn_k_b.weight");
            Ly.v_b = b16("attn_v_b.weight");
        }
        if (!Ly.dsv4) Ly.out = q("attn_output.weight");
        }
        if (Ly.moe) {
            // hy_v3: BF16 when the pack holds it so (iq_pack.py: an F32 router of exact BF16 values), else F32
            if (Ly.gqa && w16_.count(P + "ffn_gate_inp.weight") == 0) Ly.router32 = f32("ffn_gate_inp.weight");
            else Ly.router = b16("ffn_gate_inp.weight");
            Ly.router16 = w16h_.count(P + "ffn_gate_inp.weight") != 0;
            if (g.dsv4() && il < g.hash_layers) {
                // deepseek4's hash-routed layers: the token's experts from the table, no selection bias
                const auto it = wi32_.find(P + "ffn_gate_tid2eid.weight");
                if (it == wi32_.end()) missing += P + "ffn_gate_tid2eid.weight (i32) ";
                else Ly.tid2eid = it->second;
            } else {
            // (llama.cpp's hy_v3 GGUFs name the selection bias exp_probs_b, ds4's and GLM's exp_probs_b.bias)
            Ly.router_bias = w_.count(P + "exp_probs_b.bias") == 0 && w_.count(P + "exp_probs_b") != 0
                                 ? f32("exp_probs_b") : f32("exp_probs_b.bias");
            }
            Ly.sh_gate = q("ffn_gate_shexp.weight");
            Ly.sh_up = q("ffn_up_shexp.weight");
            Ly.sh_down = q("ffn_down_shexp.weight");
            const auto& nl = pack_layers_[(size_t) il];
            if (nl.layer < 0) {
                missing += P + "native experts ";
            } else {
                Ly.gu_type = nl.fmt.gu_type;
                Ly.d_type = nl.fmt.d_type;
                Ly.gu_bytes = (size_t) nl.fmt.gu_row * (size_t) g.n_ff_exp;
                Ly.dn_bytes = (size_t) nl.fmt.d_row * (size_t) g.n_embd;
                Ly.down_off = 2 * Ly.gu_bytes;
                Ly.blob = 2 * Ly.gu_bytes + Ly.dn_bytes;
                if (!gf::moe_supported(Ly.gu_type) || !gf::moe_supported(Ly.d_type) ||
                    gf::row_bytes(Ly.gu_type, g.n_embd) != nl.fmt.gu_row ||
                    gf::row_bytes(Ly.d_type, g.n_ff_exp) != nl.fmt.d_row)
                    missing += P + "expert types " + std::to_string(Ly.gu_type) + "/" + std::to_string(Ly.d_type) + " ";
            }
        } else {
            Ly.ffn_gate = q("ffn_gate.weight");
            Ly.ffn_up = q("ffn_up.weight");
            Ly.ffn_down = q("ffn_down.weight");
        }
    }
    if (!missing.empty()) {
        err = "glm fast: unresolved weights: " + missing.substr(0, 600);
        return false;
    }

    // ---- the activation arena
    const int64_t E = g.n_embd, DI = g.d_inner(), FF = g.n_ff_exp * g.n_shared;
    const int64_t max_pools = std::max<int64_t>(1, max_ctx_ / g.idx_kpool);
    const size_t q8 = 36;   // bytes per q8_1 block
    size_t need = 0;
    {
        // generous upper bound, carved below
        need = (size_t) (4 * E + 64 + 32 * 26 + 128 + 6 * DI + 4 * 128 + 2 * DI + 2 * g.q_lora + 2 * g.kv_lora +
                         4 * g.idx_key + 64 + (int64_t) g.n_head * g.qk_nope + (int64_t) g.idx_heads * g.idx_key +
                         max_pools + g.n_sel_max() + 2 * g.n_expert + 2 * FF + 2 * g.n_ff_dense + 2 * E + 64) *
                   sizeof(float);
        need += (size_t) (E + DI + g.q_lora + (int64_t) g.n_head * g.v_head + FF + 8 * g.n_ff_exp + g.n_ff_dense + E) /
                32 * q8;
        need += 64 * 256 + (size_t) (2 * NL * g.n_expert + NL * gf::kSpares) * sizeof(unsigned long long) +
                8 * 16 + 4096 * 4 + 4 * 256 + (size_t) g.n_expert * 4 + 256 + 4 * 256;
        need += (size_t) gf::kAhead * g.n_expert * sizeof(float) + 256 + (size_t) E * sizeof(float) + 256;
    }
    if (cudaMalloc(&F->arena, need) != cudaSuccess) {
        err = "glm fast: the activation arena did not allocate";
        return false;
    }
    cudaMemset(F->arena, 0, need);
    F->arena_bytes = need;
    uint8_t* at = (uint8_t*) F->arena;
    F->x = fa_take(at, E);
    F->mixer = fa_take(at, E);
    F->ffn = fa_take(at, E);
    F->pre = fa_take(at, 8);
    F->post = fa_take(at, 8);
    F->comb = fa_take(at, 16);
    F->part = fa_take(at, (size_t) (4 * E / 512) * 26 + 64);   // 25 partials per block + the norm's per-block sums
    F->counter = (unsigned int*) fa_take_b(at, 64);
    F->xq = fa_take_b(at, (size_t) E / 32 * q8);
    for (int i = 0; i < 3; ++i) F->proj[i] = fa_take(at, DI);
    for (int i = 0; i < 3; ++i) F->conv[i] = fa_take(at, DI);
    F->fa = fa_take(at, g.kda_head_dim);
    F->ga = fa_take(at, g.kda_head_dim);
    F->beta = fa_take(at, g.n_head);
    F->g1 = fa_take(at, DI);
    F->g2 = fa_take(at, DI);
    F->gated_q = fa_take_b(at, (size_t) DI / 32 * q8);
    F->qr_raw = fa_take(at, g.q_lora);
    F->qr = fa_take(at, g.q_lora);
    F->qr_q = fa_take_b(at, (size_t) g.q_lora / 32 * q8);
    F->kv_raw = fa_take(at, g.kv_lora);
    F->ik_raw = fa_take(at, g.idx_key);
    F->ig_raw = fa_take(at, g.idx_key);
    F->iw = fa_take(at, g.idx_heads);
    F->q = fa_take(at, (size_t) g.n_head * g.qk_nope);
    F->iq = fa_take(at, (size_t) g.idx_heads * g.idx_key);
    F->score = fa_take(at, (size_t) max_pools);
    F->cells = (int*) fa_take(at, (size_t) g.n_sel_max());
    F->attn_q = fa_take_b(at, (size_t) g.n_head * g.v_head / 32 * q8);
    F->rlog = fa_take(at, g.n_expert);
    F->plog = fa_take(at, g.n_expert);
    F->alog = fa_take(at, (size_t) gf::kAhead * g.n_expert);
    F->sh_out = fa_take(at, E);
    F->sh_g = fa_take(at, FF);
    F->sh_u = fa_take(at, FF);
    F->sh_hq = fa_take_b(at, (size_t) FF / 32 * q8);
    F->hq = fa_take_b(at, (size_t) 8 * g.n_ff_exp / 32 * q8);
    F->dg = fa_take(at, g.n_ff_dense);
    F->du = fa_take(at, g.n_ff_dense);
    F->dhq = fa_take_b(at, (size_t) g.n_ff_dense / 32 * q8);
    F->head_x = fa_take(at, E);
    F->head_xq = fa_take_b(at, (size_t) E / 32 * q8);
    F->emb = fa_take(at, E);
    F->tok = (int*) fa_take_b(at, 64);
    F->n_keys = NL * g.n_expert;
    F->tab = (unsigned long long*) fa_take_b(
        at, (size_t) (2 * F->n_keys + NL * gf::kSpares) * sizeof(unsigned long long));
    F->md.tab = F->tab;
    F->md.n_keys = F->n_keys;
    F->md.scratch = (unsigned long long*) fa_take_b(at, 8 * sizeof(unsigned long long));
    F->md.fetch_src = (unsigned long long*) fa_take_b(at, 8 * sizeof(unsigned long long));
    for (int b2 = 0; b2 < 2; ++b2) {
        F->pf_buf[b2] = (unsigned long long*) fa_take_b(at, 16 * sizeof(unsigned long long));
        F->pf_n_buf[b2] = (int*) fa_take_b(at, 64);
    }
    F->md.plan_ptr = (unsigned long long*) fa_take_b(at, 8 * sizeof(unsigned long long));
    F->md.plan_w = fa_take(at, 8);
    F->md.plan_id = (int*) fa_take(at, 8);
    F->md.seq = (unsigned int*) fa_take_b(at, 64);
    F->md.wait_seq = (unsigned int*) fa_take_b(at, 64);
    F->md.cpu_part = fa_take(at, E);
    F->md.cpu_flag = (int*) fa_take_b(at, 64);
    if ((size_t) (at - (uint8_t*) F->arena) > need) {
        err = "glm fast: the activation arena overflowed its estimate";
        return false;
    }
    if (g.gqa()) {
        const size_t qf = (size_t) g.n_head * g.head_dim, kf = (size_t) g.n_kv * g.head_dim;
        const int max_keys = (int) std::max<int64_t>(max_ctx_, kvs_ != nullptr ? (int64_t) kvs_->sel.max_sel * gf::kGqaPage : 0);
        const size_t sf = gf::gqa_attn_scratch_floats(g.n_head, max_keys);
        if (cudaMalloc(&F->gq, (qf + 2 * kf + sf) * sizeof(float)) != cudaSuccess) {
            err = "glm fast: the attention buffers did not allocate";
            return false;
        }
        F->gk = F->gq + qf;
        F->gv = F->gk + kf;
        F->gscr = F->gv + kf;
    }
    if (g.dsv4()) {
        // deepseek4's attention buffers (floats; the q8_1 copies at 36 / 32 bytes a value, rounded up)
        const size_t QL = (size_t) g.q_lora, QH = (size_t) g.n_head * g.head_dim, LO = (size_t) g.o_groups * g.o_lora,
                     IQ = (size_t) g.idx_heads * g.idx_key, NC = (size_t) std::max<int64_t>(1, max_ctx_ / 4 + 1);
        const auto q8f = [](size_t n) { return (n / 32 * 36 + 3) / 4; };
        const size_t fl = 2 * QL + q8f(QL) + QH + (size_t) g.head_dim + 4 * (size_t) g.head_dim + 4 * (size_t) g.idx_key +
                          IQ + (size_t) g.idx_heads + NC + QH + q8f(QH) + LO + q8f(LO) + (size_t) g.idx_top_k + 64 * 16;
        if (cudaMalloc(&F->d4, fl * sizeof(float)) != cudaSuccess) {
            err = "glm fast: the deepseek4 attention buffers did not allocate";
            return false;
        }
        float* at = F->d4;
        const auto take = [&](size_t n) { float* r = at; at += (n + 15) / 16 * 16; return r; };
        F->d4_qr = take(QL);
        F->d4_qrn = take(QL);
        F->d4_qrn_q8 = take(q8f(QL));
        F->d4_q = take(QH);
        F->d4_kv = take((size_t) g.head_dim);
        F->d4_ckv = take(2 * (size_t) g.head_dim);
        F->d4_csc = take(2 * (size_t) g.head_dim);
        F->d4_ickv = take(2 * (size_t) g.idx_key);
        F->d4_icsc = take(2 * (size_t) g.idx_key);
        F->d4_iq = take(IQ);
        F->d4_iw = take((size_t) g.idx_heads);
        F->d4_score = take(NC);
        F->d4_heads = take(QH);
        F->d4_heads_q8 = take(q8f(QH));
        F->d4_low = take(LO);
        F->d4_low_q8 = take(q8f(LO));
        F->d4_sel = (int*) take((size_t) g.idx_top_k);
        // each layer's rope (rope_tail_layer_inplace): the compressed layers on compress_rope_freq_base with YaRN
        F->d4_rope.assign((size_t) g.n_layers + 1, gf::Dsv4Rope{});
        for (int il = 0; il < g.n_layers; ++il) {
            gf::Dsv4Rope& r = F->d4_rope[(size_t) il];
            const bool comp = g.ratio(il) != 0;
            r.base = comp && g.comp_rope_base > 0.0f ? g.comp_rope_base : g.rope_base;
            r.scale = comp && g.rope_factor > 0.0f ? 1.0f / g.rope_factor : 1.0f;
            r.ext = comp && g.rope_factor > 1.0f ? 1.0f : 0.0f;
            if (r.ext != 0.0f && g.rope_orig_ctx > 0) {
                const float nd = (float) g.n_rot;
                const auto cd = [&](float beta) {
                    return nd * std::log((float) g.rope_orig_ctx / (beta * 2.0f * (float) M_PI)) / (2.0f * std::log(r.base));
                };
                r.corr0 = std::max(0.0f, std::floor(cd(g.yarn_fast)));
                r.corr1 = std::min(nd - 1.0f, std::ceil(cd(g.yarn_slow)));
            }
        }
    }
    if (mtp_il_ >= 0) {
        if (cudaMalloc(&F->mtp_h, (size_t) g.n_embd * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&F->mtp_catq, (size_t) 2 * g.n_embd / 32 * 36) != cudaSuccess ||
            cudaMalloc(&F->mtp_logits, (size_t) g.n_vocab * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&F->mtp_tok, 64) != cudaSuccess ||
            cudaHostAlloc((void**) &F->mtp_tok_h, 64, cudaHostAllocDefault) != cudaSuccess ||
            cudaEventCreateWithFlags(&F->ev_mtp, cudaEventDisableTiming) != cudaSuccess) {
            err = "glm fast: the NextN block's buffers did not allocate";
            return false;
        }
    }

    // ---- host-mapped routing ring + response, pinned staging for the embedding / hop / token
    void* hp = nullptr;
    if (cudaHostAlloc(&hp, sizeof(gf::MoeRequest) * gf::kRingSize + sizeof(int), cudaHostAllocMapped) != cudaSuccess) {
        err = "glm fast: the routing ring did not allocate";
        return false;
    }
    std::memset(hp, 0, sizeof(gf::MoeRequest) * gf::kRingSize + sizeof(int));
    F->ring_h = (gf::MoeRequest*) hp;
    void* dp = nullptr;
    cudaHostGetDevicePointer(&dp, hp, 0);
    F->md.ring = dp;
    F->route_error_h = (volatile int*) (F->ring_h + gf::kRingSize);
    F->md.route_error = (volatile int*) ((gf::MoeRequest*) dp + gf::kRingSize);
    if (cudaHostAlloc(&hp, sizeof(gf::MoeResponse), cudaHostAllocMapped) != cudaSuccess) {
        err = "glm fast: the routing response did not allocate";
        return false;
    }
    std::memset(hp, 0, sizeof(gf::MoeResponse));
    F->resp_h = (gf::MoeResponse*) hp;
    cudaHostGetDevicePointer(&dp, hp, 0);
    F->md.resp = dp;
    if (cudaHostAlloc((void**) &F->emb_h, (size_t) E * sizeof(float), cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc((void**) &F->hop_h, (size_t) g.hc * E * sizeof(float), cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc((void**) &F->tok_h, 64, cudaHostAllocDefault) != cudaSuccess) {
        err = "glm fast: pinned staging did not allocate";
        return false;
    }

    // ---- the expert pool: per-layer partitions (equal quota per layer within its size class)
    size_t blob_max = 0;
    int n_moe = 0;
    for (int il = l0_; il < lt_; ++il)
        if (F->L[(size_t) il].moe) {
            blob_max = std::max(blob_max, F->L[(size_t) il].blob);
            ++n_moe;
        }
    F->lp.assign((size_t) NL, FastState::LayerPool{});
    F->slot_of.assign((size_t) NL * g.n_expert, -1);
    if (n_moe > 0) {
        // the scratch slots: a fetched expert that found no spare lands here (8 = one per routed entry)
        const size_t sstride = (blob_max + 255u) & ~(size_t) 255u;
        if (cudaMalloc(&F->scratch, 8 * sstride) != cudaSuccess) {
            err = "glm fast: the scratch slots did not allocate";
            return false;
        }
        {
            unsigned long long sp[8];
            for (int i = 0; i < 8; ++i) sp[i] = (unsigned long long) (F->scratch + (size_t) i * sstride);
            cudaMemcpy(F->md.scratch, sp, sizeof sp, cudaMemcpyHostToDevice);
        }
        // a parallel split load: the parts pin their prompt paths, size their pools and plan their RAM tiers one at a
        // time, in part order - the first part's RAM measurement then sees none of the later parts' prompt staging, as
        // when they load one after another (it took that staging twice: once pinned, once in the headroom)
        if (split_load_ != nullptr) split_load_->wait(part_);
        // the batched prompt path: its sizes (it borrows the pool's tail, see below).  (deepseek4: none yet - its
        // prompts run through the token path)
        if (!prefill_setup(err)) return false;
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        {
            const double G = 1073741824.0, used = (double) (total_b - free_b);
            const double known = (double) (w_arena_bytes_ + state_bytes_ + sc_bytes_ + F->arena_bytes + 8 * sstride);
            std::fprintf(stderr, "glm fast: CUDA%d VRAM before the expert pool: %.2f of %.2f GB in use - dense weights %.2f, "
                                 "state/KV (%lld ctx) %.2f, activations %.2f, other (context, MTP, scratch, prompt path) "
                                 "%.2f\n", dev_, used / G, (double) total_b / G, (double) w_arena_bytes_ / G,
                         (long long) max_ctx_, (double) state_bytes_ / G,
                         (double) (sc_bytes_ + F->arena_bytes + 8 * sstride) / G, (used - known) / G);
        }
        // headroom: cuBLAS-free path; the context's own growth is already allocated (state arena);
        // keep ~700 MB for the driver, the sampler and kernel launches' local memory
        size_t reserve = (size_t) (F->unified_memory ? 1024 : 700) << 20;
        if (const char* r = getenv("STRATA_GLM_RESERVE_MB")) reserve = (size_t) std::atoll(r) << 20;
        size_t avail = glmfast::expert_pool_budget(false, free_b, 0, reserve, 0, 0);
#if defined(STRATA_USE_HIP)
        if (F->unified_memory) {
            // Measure after dense/KV/scratch and pinned prompt staging are allocated.
            // Prompt device buffers borrow the pool, so their budget is not subtracted again.
            long long avail_kb = 0;
            if (FILE* mf = std::fopen("/proc/meminfo", "r")) {
                char line[256];
                while (std::fgets(line, sizeof line, mf))
                    if (std::sscanf(line, "MemAvailable: %lld kB", &avail_kb) == 1) break;
                std::fclose(mf);
            }
            if (avail_kb > 0) avail_kb += (long long) (glmfast::zfs_arc_reclaimable() >> 10);
            double head_gb = 16.0;
            if (const char* h = getenv("STRATA_GLM_RAM_HEADROOM_GB")) head_gb = std::max(0.0, std::atof(h));
            size_t expert_bytes = 0;
            for (int il = l0_; il < lt_; ++il) {
                const auto& Ly = F->L[(size_t) il];
                if (Ly.moe) expert_bytes += (size_t) g.n_expert * glmfast::expert_stride(Ly.blob, Ly.gu_type, Ly.d_type);
            }
            avail = glmfast::expert_pool_budget(true, free_b, (size_t) std::max(0LL, avail_kb) * 1024,
                                                reserve, (size_t) (head_gb * 1073741824.0), expert_bytes);
            std::fprintf(stderr, "glm fast: HIP%d unified memory: MemAvailable %.2f GB, headroom %.2f GB + "
                                 "%zu MB reserve, experts need %.2f GB, pool budget %.2f GB\n", dev_,
                         (double) avail_kb / 1048576.0, head_gb, reserve >> 20,
                         (double) expert_bytes / 1073741824.0, (double) avail / 1073741824.0);
        }
#endif
        if (const char* cap = getenv("STRATA_GLM_POOL_GB"))
            avail = std::min(avail, (size_t) (std::atof(cap) * 1073741824.0));
        // STRATA_GLM_VRAM_GB=<n>: behave like a card with n GB - the cap counts everything this process already
        // holds on the device (context, dense weights, state, arenas), so the pool gets what such a card would have
        if (const char* vc = getenv("STRATA_GLM_VRAM_GB")) {
            const size_t cap = (size_t) (std::atof(vc) * 1073741824.0);
            const size_t used = total_b - free_b;
            const size_t room = cap > used + reserve ? cap - used - reserve : 0;
            std::fprintf(stderr, "glm fast: CUDA%d VRAM cap %.1f GB: %.2f GB already in use, %.2f GB left for experts\n",
                         dev_, (double) cap / 1073741824.0, (double) used / 1073741824.0, (double) room / 1073741824.0);
            avail = std::min(avail, room);
        }
        // deepseek4 with its MTP block (STRATA_DSV4_MTP): the block's weights (its experts too) stay out of the pool
        if (const char* mp = getenv("STRATA_DSV4_MTP"); mp != nullptr && mp[0] && g.dsv4()) {
            std::error_code ec;
            size_t keep = (size_t) std::filesystem::file_size(mp, ec) + ((size_t) 96 << 20);
            if (const int et = d4_mtp_exp_type(); !ec && et >= 0) {   // the experts requantized at load
                try {
                    strata::GgufFile mf(mp);
                    for (const char* n : {"mtp.0.ffn_gate_exps.weight", "mtp.0.ffn_up_exps.weight", "mtp.0.ffn_down_exps.weight"})
                        if (const strata::TensorInfo* t = mf.find(n)) {
                            const size_t rows = t->elements() / t->shape[0];
                            keep -= ggml_row_size((ggml_type) t->type, (int64_t) t->shape[0]) * rows;
                            keep += ggml_row_size((ggml_type) et, (int64_t) t->shape[0]) * rows;
                        }
                } catch (...) {
                }
            }
            if (const char* r = getenv("STRATA_DSV4_MTP_RESERVE_GB")) keep = (size_t) (std::atof(r) * 1073741824.0);
            if (!ec || getenv("STRATA_DSV4_MTP_RESERVE_GB")) {
                avail = avail > keep ? avail - keep : 0;
                std::fprintf(stderr, "glm fast: CUDA%d %.2f GB kept out of the expert pool for the MTP block\n", dev_,
                             (double) keep / 1073741824.0);
            }
        }
        // a layer's slot stride: MMQ-addressable (glmfast::expert_stride), so the prompt path multiplies the pool
        // partition in place
        const auto slot_stride = [&](int il) {
            const auto& Ly = F->L[(size_t) il];
            return glmfast::expert_stride(Ly.blob, Ly.gu_type, Ly.d_type);
        };
        // every layer gets the same number of slots: avail / sum(blob over layers)
        size_t blob_sum = 0;
        for (int il = l0_; il < lt_; ++il)
            if (F->L[(size_t) il].moe) blob_sum += F->L[(size_t) il].blob;
        int per = (int) std::min<size_t>((size_t) g.n_expert, avail / std::max<size_t>(1, blob_sum));
        // keep a whole number of 256-byte-aligned blobs
        while (per > 0) {
            size_t tot = 0;
            for (int il = l0_; il < lt_; ++il)
                if (F->L[(size_t) il].moe) tot += (size_t) per * slot_stride(il);
            if (tot <= avail) break;
            --per;
        }
        if (per < g.n_exp_used) {
            err = "glm fast: not enough VRAM for the expert pool (" + std::to_string(per) + " slots per layer)";
            return false;
        }
        size_t stride_sum = 0;
        for (int il = l0_; il < lt_; ++il)
            if (F->L[(size_t) il].moe) stride_sum += slot_stride(il);
        // the prompt path borrows the pool's TAIL: the last k slots of every layer, laid out as one region after all
        // the main segments - the prompt path's buffers while a prompt runs, expert slots the rest of the time
        int k_extra = 0;
        const auto tail_slots = [&](size_t bytes) {
            return (int) ((bytes + stride_sum - 1) / std::max<size_t>(1, stride_sum));
        };
        // hy_v3: region b of the K / V records is the tail's last kv_b_bytes_ - slots of the decode until the context
        // reaches them (kv_ensure); the prompt path's and the vision encoder's needs are sized from the slots before it
        const int k_kv = kv_b_bytes_ > 0 ? tail_slots(kv_b_bytes_ + 256) : 0;
        if (k_kv > 0 && per - k_kv < g.n_exp_used + gf::kSpares + 2) {
            err = "glm fast: the expert pool (" + std::to_string(per) + " slots/layer) cannot hold the K / V cache's " +
                  std::to_string(kv_b_bytes_ >> 20) + " MB beyond position " + std::to_string(kv_p0_) +
                  " - a smaller context or STRATA_GLM_KV=k8v4";
            return false;
        }
        if (pf_ != nullptr) {
            k_extra = tail_slots(prefill_borrow_bytes());
            // (the prestage buffer is the part that can shrink: to what the pool can lend, else none)
            const int k_max = per - (g.n_exp_used + gf::kSpares + 2);
            if (k_extra > k_max && k_max > 0) k_extra = tail_slots(prefill_trim_prestage((size_t) k_max * stride_sum));
            // (a prompt runs alone - no decode beside it, unlike the vision encoder's lending below - so its main
            // slots need little more than one route's experts and the spares)
            if (per - k_extra < g.n_exp_used + gf::kSpares + 2) {
                std::fprintf(stderr, "glm prefill: CUDA%d the pool (%d slots/layer) cannot lend %zu MB - token by token\n",
                             dev_, per, prefill_borrow_bytes() >> 20);
                prefill_destroy();
                k_extra = 0;
            }
        }
        // an on-demand vision encoder on the first GPU (the server sets STRATA_GLM_VISION_LEND_MB) borrows the same
        // tail while it encodes, so the tail there is the larger of the two needs
        vis_lend_ok_ = false;
        if (const char* v = getenv("STRATA_GLM_VISION_LEND_MB"); v != nullptr && dev_ == 0 && k_kv == 0) {
            const size_t vb = (size_t) std::max(0LL, std::atoll(v)) << 20;
            if (vb > 0 && per - tail_slots(vb) >= g.n_exp_used + gf::kSpares + 8) {
                k_extra = std::max(k_extra, tail_slots(vb));
                vis_lend_ok_ = true;
            } else if (vb > 0) {
                std::fprintf(stderr, "glm fast: CUDA%d the pool (%d slots/layer) cannot lend %zu MB to the vision "
                                     "encoder\n", dev_, per, vb >> 20);
            }
        }
        // (the two share the tail: the records fill it from the top down, the prompt path's buffers from the bottom up,
        // and a prompt's chunk shrinks to the room below the records it needs - prefill().  The tail holds both whole
        // when the main slots keep their floor: a tail slot is an expert slot whenever it is not lent, so the decode
        // loses nothing by it)
        if (k_kv > 0)
            k_extra = std::max(std::max(k_extra, k_kv),
                               std::min(k_extra + k_kv, per - (g.n_exp_used + gf::kSpares + 2)));
        // the slots PER LAYER: uniform, or - with the pack's expert_counts.txt (tools/glm_expert_prior.py) - by routing
        // share: every layer gets a floor, then each further slot goes to the layer whose next-most-routed expert serves
        // the largest share of its lookups (early layers spread their routing wide and gain the most).
        // STRATA_GLM_UNIFORM_SLOTS=1 keeps the uniform split (A/B).
        std::vector<int> nsl((size_t) NL, 0);
        for (int il = l0_; il < lt_; ++il)
            if (F->L[(size_t) il].moe) nsl[(size_t) il] = per;
        // When the APU can hold every expert, keep exactly that many slots in each
        // layer rather than spending the capped budget on uneven partitions/spares.
        if (getenv("STRATA_GLM_UNIFORM_SLOTS") == nullptr && !(F->unified_memory && per == g.n_expert)) {
            std::vector<std::vector<double>> share((size_t) NL);
            std::ifstream cf(pack_dir_ + "/expert_counts.txt");
            std::string line;
            while (std::getline(cf, line)) {
                std::istringstream ss(line);
                int il = -1;
                ss >> il;
                if (il < l0_ || il >= lt_ || !F->L[(size_t) il].moe) continue;
                std::vector<double> c;
                double v = 0, tot_c = 0;
                while (ss >> v) {
                    c.push_back(v);
                    tot_c += v;
                }
                if (tot_c <= 0) continue;
                std::sort(c.rbegin(), c.rend());
                for (double& x : c) x /= tot_c;
                share[(size_t) il] = std::move(c);
            }
            // a layer the counts do not cover (the NextN block) takes the mean profile of the others
            {
                std::vector<double> mean;
                int nm = 0;
                for (int il = l0_; il < lt_; ++il) {
                    const auto& sh = share[(size_t) il];
                    if (sh.empty()) continue;
                    if (mean.size() < sh.size()) mean.resize(sh.size(), 0.0);
                    for (size_t j = 0; j < sh.size(); ++j) mean[j] += sh[j];
                    ++nm;
                }
                if (nm > 0) {
                    for (double& x : mean) x /= nm;
                    for (int il = l0_; il < lt_; ++il)
                        if (F->L[(size_t) il].moe && share[(size_t) il].empty()) share[(size_t) il] = mean;
                }
            }
            bool all = true;
            for (int il = l0_; il < lt_; ++il)
                if (F->L[(size_t) il].moe && share[(size_t) il].empty()) all = false;
            if (all) {
                const int cap = g.n_expert + gf::kSpares;
                // the floor: a route's experts, the spares and the lendable tail, +16 when the budget has room for
                // it (a 24 GB card running the whole model has ~30 slots a layer: the +16 floor alone overran it)
                const int need = g.n_exp_used + gf::kSpares + k_extra;
                const int fl = std::min(cap, std::max(need, std::min(per, need + 16)));
                size_t used = 0;
                for (int il = l0_; il < lt_; ++il)
                    if (F->L[(size_t) il].moe) {
                        nsl[(size_t) il] = fl;
                        used += (size_t) fl * slot_stride(il);
                    }
                // the value of layer l's next slot: the routing share of its next expert (spares past the 288th serve none)
                const auto gain = [&](int il) {
                    const auto& sh = share[(size_t) il];
                    const int j = nsl[(size_t) il] - gf::kSpares;
                    return (j >= 0 && j < (int) sh.size()) ? sh[(size_t) j] : 0.0;
                };
                for (;;) {
                    int best = -1;
                    double bg = -1.0;
                    for (int il = l0_; il < lt_; ++il) {
                        if (!F->L[(size_t) il].moe || nsl[(size_t) il] >= cap) continue;
                        if (used + slot_stride(il) > avail) continue;
                        const double gv = gain(il);
                        if (gv > bg) {
                            bg = gv;
                            best = il;
                        }
                    }
                    if (best < 0) break;
                    ++nsl[(size_t) best];
                    used += slot_stride(best);
                }
            }
        }
        size_t tot = 0, xtot = 0;
        int nmin = INT32_MAX, nmax = 0;
        for (int il = l0_; il < lt_; ++il)
            if (F->L[(size_t) il].moe) {
                tot += (size_t) nsl[(size_t) il] * slot_stride(il);
                xtot += (size_t) k_extra * slot_stride(il);
                nmin = std::min(nmin, nsl[(size_t) il]);
                nmax = std::max(nmax, nsl[(size_t) il]);
            }
        // the main slots in one allocation - or, where the driver caps a single allocation below them (Windows gives
        // a 160 GiB APU carve-out in blocks of at most ~96 GiB), in several, each holding whole layers: a layer's
        // slots are addressed from its own base (P.base), never across layers
        std::vector<uint8_t*> segs;
        std::vector<int> seg_of((size_t) NL, 0);
        const size_t main_bytes = tot - xtot;
        if (uint8_t* one = nullptr; cudaMalloc(&one, main_bytes) == cudaSuccess) {
            segs.push_back(one);
        } else {
            cudaGetLastError();
            size_t layer_max = 0;
            for (int il = l0_; il < lt_; ++il)
                if (F->L[(size_t) il].moe)
                    layer_max = std::max(layer_max, (size_t) (nsl[(size_t) il] - k_extra) * slot_stride(il));
            for (size_t cap = main_bytes / 2; segs.empty() && cap >= layer_max && layer_max > 0; cap /= 2) {
                std::vector<size_t> sizes(1, 0);
                for (int il = l0_; il < lt_; ++il) {
                    if (!F->L[(size_t) il].moe) continue;
                    const size_t lb = (size_t) (nsl[(size_t) il] - k_extra) * slot_stride(il);
                    if (sizes.back() > 0 && sizes.back() + lb > cap) sizes.push_back(0);
                    sizes.back() += lb;
                    seg_of[(size_t) il] = (int) sizes.size() - 1;
                }
                for (size_t sz : sizes) {
                    uint8_t* p = nullptr;
                    if (cudaMalloc(&p, sz) != cudaSuccess) {
                        cudaGetLastError();
                        for (uint8_t* q : segs) cudaFree(q);
                        segs.clear();
                        break;
                    }
                    segs.push_back(p);
                }
            }
            if (!segs.empty())
                std::fprintf(stderr, "glm fast: CUDA%d one allocation of %.2f GB did not allocate: the expert pool is in "
                                     "%zu\n", dev_, (double) main_bytes / 1073741824.0, segs.size());
        }
        if (segs.empty() || (xtot > 0 && cudaMalloc(&F->xpool, xtot) != cudaSuccess)) {
            for (uint8_t* q : segs) cudaFree(q);
            err = "glm fast: the expert pool (" + std::to_string(tot >> 20) + " MB) did not allocate";
            return false;
        }
        F->pool = segs[0];
        F->pool_more.assign(segs.begin() + 1, segs.end());
        F->pool_bytes = tot;
        F->xpool_bytes = xtot;
        uint8_t* b = F->pool;
        int cur_seg = 0;
        for (int il = l0_; il < lt_; ++il) {
            if (!F->L[(size_t) il].moe) continue;
            auto& P = F->lp[(size_t) il];
            const int nl = nsl[(size_t) il];
            if (seg_of[(size_t) il] != cur_seg) b = segs[(size_t) (cur_seg = seg_of[(size_t) il])];
            P.stride = slot_stride(il);
            P.n = nl;
            P.n_main = nl - k_extra;
            P.base = b;
            P.key.assign((size_t) nl, -1);
            P.tick.assign((size_t) nl, 0);
            P.st.assign((size_t) nl, FastState::kFree);
            b += (size_t) P.n_main * P.stride;
            F->pool_slots += nl;
        }
        uint8_t* xregion = F->xpool;
        b = xregion;
        for (int il = l0_; il < lt_; ++il) {
            if (!F->L[(size_t) il].moe) continue;
            auto& P = F->lp[(size_t) il];
            P.xbase = b;
            b += (size_t) k_extra * P.stride;
        }
        if (k_kv > 0) {
            kv_b_ = F->xpool + xtot - kv_rec_;   // (record p0: the top one)
            kv_lent_ = kv_p0_;
            std::fprintf(stderr, "glm fast: CUDA%d the pool's last %.2f GB (%d slots/layer) hold the K / V records past "
                                 "position %lld as the context reaches them\n", dev_, (double) kv_b_bytes_ / 1073741824.0,
                         k_kv, (long long) kv_p0_);
        }
        if (pf_ != nullptr && !prefill_bind(xregion, xtot, err)) return false;
        if (vis_lend_ok_)
            std::fprintf(stderr, "glm fast: CUDA%d the vision encoder borrows the pool's tail (%.2f GB) while it "
                                 "encodes\n", dev_, (double) xtot / 1073741824.0);
        F->cnt.assign((size_t) NL * g.n_expert, 0);
        F->usage.assign((size_t) NL * g.n_expert, 0);
        F->pred_of.assign((size_t) NL, std::array<int, 8>{-1, -1, -1, -1, -1, -1, -1, -1});
        std::fprintf(stderr, "glm fast: CUDA%d layers [%d,%d) expert pool %.2f GB, %d-%d slots/layer (%lld total)\n",
                     dev_, l0_, l1_, (double) tot / 1073741824.0, nmin, nmax, (long long) F->pool_slots);
        if (getenv("STRATA_GLM_TIMING") != nullptr) {
            std::string sl;
            for (int il = l0_; il < lt_; ++il)
                if (F->L[(size_t) il].moe) sl += std::to_string(il) + ":" + std::to_string(nsl[(size_t) il]) + " ";
            std::fprintf(stderr, "glm fast: CUDA%d slots per layer %s\n", dev_, sl.c_str());
        }

        // the model's clean page cache goes first (once): the experts are read with O_DIRECT, so it never serves the
        // engine, and on a 2-socket host it fills one node - MPOL_INTERLEAVE then falls back instead of reclaiming
        // (49 GB of the GGUF cached on node 1 left a 100 GB tier 78% on node 0: the CPU lane read one node's
        // controllers, ~60 instead of ~92 GB/s).  Pages some process still maps stay.  STRATA_GLM_DROP_CACHE=0 keeps it.
#ifdef __linux__   // (numa_nodes() is empty elsewhere; posix_fadvise is POSIX)
        {
            static std::atomic<bool> dropped{false};
            const char* dc = getenv("STRATA_GLM_DROP_CACHE");
            if ((dc == nullptr || std::atoi(dc) != 0) && numa_nodes().size() >= 2 && !dropped.exchange(true)) {
                for (const Shard& sh : pack_shards_)
                    if (sh.fd >= 0) posix_fadvise(sh.fd, 0, 0, POSIX_FADV_DONTNEED);
            }
        }
#endif
        // ---- the RAM tier: one pinned arena per blob size class, slots in proportion to the class's
        //      share of this half's expert bytes NOT already held by the VRAM pool
        F->layer_rc.assign((size_t) NL, -1);
        F->ram_of.assign((size_t) NL * g.n_expert, -1);
        F->left.assign((size_t) NL * g.n_expert, 0);
        std::vector<size_t> cls_stride;
        std::vector<double> cls_weight;
        for (int il = l0_; il < lt_; ++il) {
            if (!F->L[(size_t) il].moe) continue;
            const size_t st = (F->L[(size_t) il].blob + 4095u) & ~(size_t) 4095u;
            int c = -1;
            for (size_t k = 0; k < cls_stride.size(); ++k)
                if (cls_stride[k] == st) c = (int) k;
            if (c < 0) {
                c = (int) cls_stride.size();
                cls_stride.push_back(st);
                cls_weight.push_back(0.0);
            }
            F->layer_rc[(size_t) il] = c;
            // (the experts its main slots cannot hold: the pool's lendable tail counts as not held - a prompt
            // borrows it, and the experts there go to RAM meanwhile, prefill_lend)
            cls_weight[(size_t) c] +=
                (double) std::max(2, g.n_expert - F->lp[(size_t) il].n_main + gf::kSpares + 2 + F->ram_slack) * (double) st;
        }
        double wsum = 0;
        for (double w : cls_weight) wsum += w;
        int64_t budget = ram_budget_;
        const bool staging_only = glmfast::minimal_ram_tier(F->unified_memory, nmin >= g.n_expert,
                                                           ram_budget_ >= 0 || getenv("STRATA_GLM_RAM_GB") != nullptr);
        if (staging_only) {
            // Decode misses and pool-tail lending still need landing slots. Keep
            // only the existing per-class floor; do not duplicate the expert cache.
            budget = 0;
            std::fprintf(stderr, "glm fast: HIP%d all experts fit in the GPU pool; RAM tier is staging only\n", dev_);
        }
        // the whole machine's budget is measured ONCE (the first half's pinning would shrink what the second
        // half sees): MemAvailable minus headroom for the OS, the server and the page cache the disk reads go
        // through; STRATA_GLM_RAM_GB pins it.  Each half takes its share of the MoE layers still unserved,
        // and whatever a half cannot use (its experts already fit) passes on to the next half.
        // The later parts' prompt path pins after this measurement (their token rows and landing ring, and the rows a
        // middle part hands on), and that came out of the headroom: one part's staging fits in it (two GPUs: 2x V100,
        // as v1.0.30), three parts' did not and froze a Linux desktop (#77).  What a third part on adds is set aside
        // here (estimated from this part's own sizes), and each later part measures again and leaves at least half
        // the headroom free.
        static int64_t total = -1, remaining = -1;
        static int remaining_layers = 0;
        static int64_t unpinned = 0;   // a parallel load: the earlier parts' planned tiers, none pinned yet
        const auto pinned_pf = prefill_pinned();
        const int later = n_parts_ - 1 - part_;   // the parts that start after this one
        const int64_t unseen_pf = (int64_t) std::max(0, later - 1) * (int64_t) pinned_pf.staging +
                                  (int64_t) std::max(0, later - 1) * (int64_t) pinned_pf.hop;
        double head_gb = F->unified_memory ? 16.0 : 6.0;
        if (const char* h = getenv("STRATA_GLM_RAM_HEADROOM_GB")) head_gb = std::max(0.0, std::atof(h));
        const int64_t head_b = (int64_t) (head_gb * 1073741824.0);
        const bool fixed_ram = getenv("STRATA_GLM_RAM_GB") != nullptr;
        if (budget < 0) {
            if (total < 0) {
                if (const char* rg = getenv("STRATA_GLM_RAM_GB")) {
                    total = (int64_t) (std::atof(rg) * 1073741824.0);
                } else {
                    int64_t avail_kb = 0;
#ifdef _WIN32
                    MEMORYSTATUSEX ms{};
                    ms.dwLength = sizeof ms;
                    // pinned RAM is charged to the commit (RAM + page file), and under WDDM so is every allocation
                    // on the card - the pool above already took its share: the smaller of the two is what can pin
                    if (GlobalMemoryStatusEx(&ms)) {
                        avail_kb = (int64_t) (std::min(ms.ullAvailPhys, ms.ullAvailPageFile) >> 10);
                        // the commit, not the RAM, caps the tier: say so (issue #20 - a page file of 8 GB on a
                        // 128 GB PC left 27 GB of RAM free while the SSD served experts)
                        const double G = 1073741824.0, short_gb = ((double) ms.ullAvailPhys - (double) ms.ullAvailPageFile) / G;
                        if (short_gb > 2.0)
                            std::fprintf(stderr, "glm fast: WARNING - Windows' commit limit (RAM + page file) caps the RAM "
                                                 "tier %.0f GB below the free RAM (%.1f GB free, %.1f GB of commit left): "
                                                 "those experts are read from the SSD. Enlarge the page file by at "
                                                 "least %.0f GB (System > Advanced system settings > Performance > "
                                                 "Virtual memory) and restart Maya\n", short_gb,
                                         (double) ms.ullAvailPhys / G, (double) ms.ullAvailPageFile / G, short_gb + 4.0);
                    }
#else
                    if (FILE* mf = std::fopen("/proc/meminfo", "r")) {
                        char line[256];
                        while (std::fgets(line, sizeof line, mf))
                            if (std::sscanf(line, "MemAvailable: %lld kB", (long long*) &avail_kb) == 1) break;
                        std::fclose(mf);
                    }
                    if (const int64_t arc = glmfast::zfs_arc_reclaimable(); arc > 0 && avail_kb > 0) {
                        avail_kb += arc >> 10;
                        std::fprintf(stderr, "glm fast: ZFS's ARC holds %.1f GB above its floor: counted as free RAM "
                                             "(it gives it back under pressure)\n", (double) arc / 1073741824.0);
                    }
#endif
                    total = std::max<int64_t>(0, avail_kb * 1024 - head_b - unseen_pf);
                }
                remaining = total;
                remaining_layers = g.n_layers - g.dense_lead + (getenv("STRATA_GLM_NO_MTP") ? 0 : g.nextn);
            }
            // the last half takes everything left (one device: the NextN block, counted above, is not loaded)
            budget = l1_ >= g.n_layers ? remaining
                                       : std::min<int64_t>(remaining, (int64_t) ((double) remaining * (double) n_moe /
                                                                                 (double) std::max(1, remaining_layers)));
            // a later part: no more than is free now, less what is still to come and half the headroom (a parallel
            // load: and less the earlier parts' tiers, planned but not pinned yet)
            if (part_ > 0 && !fixed_ram) {
                const int64_t now = avail_ram_now();
                if (now > 0)
                    budget = std::min<int64_t>(budget, std::max<int64_t>(0, now - unpinned - head_b / 2 - unseen_pf));
            }
        }
        F->rc.assign(cls_stride.size(), FastState::RamClass{});
        std::vector<int64_t> cls_n(cls_stride.size(), 0), cls_cap(cls_stride.size(), 0);
        for (size_t c = 0; c < cls_stride.size() && wsum > 0; ++c) {
            int64_t n = (int64_t) ((double) budget * cls_weight[c] / wsum / (double) cls_stride[c]);
            n = std::max<int64_t>(n, 16);   // a floor so a miss always has somewhere to land
            // never more than the experts of the class that the VRAM pool does not hold, plus the free slots
            // fast_boundary keeps per class for disk reads and reads ahead.  When the pool holds every expert (a big
            // card: 288 of 288 slots a layer, or a split across many GPUs) that is a few spares' worth - below the floor
            // for a class of one layer (the NextN block), which then never allocated and stopped the start ("did not
            // allocate")
            cls_cap[c] = std::max<int64_t>(1, (int64_t) (cls_weight[c] / (double) cls_stride[c])) + 16;
            cls_n[c] = std::min<int64_t>(n, cls_cap[c]);
        }
        // parallel load: take the planned size from the budget and let the next part plan; every part pins once all
        // have planned (a later part's measurement then sees no tier half pinned), all of them at once
        if (split_load_ != nullptr) {
            int64_t planned = 0;
            for (size_t c = 0; c < cls_n.size(); ++c) planned += cls_n[c] * (int64_t) cls_stride[c];
            if (!staging_only && ram_budget_ < 0 && remaining >= 0) {
                remaining = std::max<int64_t>(0, remaining - planned);
                remaining_layers -= n_moe;
            }
            unpinned += planned;
            split_load_->pass(part_);
            split_load_->wait(n_parts_);
        }
        // the small pinned buffers first - the table updates and the disk reads' staging (an expert a routed entry):
        // where the driver caps what can be pinned (Windows: an RTX 5090 Laptop with 64 GB, the cap reached at a 45.3
        // GB tier, #67) the tier below shrinks to fit instead of the start failing on them after it
        if (cudaHostAlloc((void**) &F->upd_key_h, FastState::kMaxUpd * sizeof(int), cudaHostAllocDefault) != cudaSuccess ||
            cudaHostAlloc((void**) &F->upd_val_h, FastState::kMaxUpd * sizeof(unsigned long long),
                          cudaHostAllocDefault) != cudaSuccess ||
            cudaMalloc((void**) &F->upd_key_d, FastState::kMaxUpd * sizeof(int)) != cudaSuccess ||
            cudaMalloc((void**) &F->upd_val_d, FastState::kMaxUpd * sizeof(unsigned long long)) != cudaSuccess) {
            err = "glm fast: the table update buffers did not allocate";
            return false;
        }
        F->stage.assign((size_t) std::max(8, g.n_exp_used), nullptr);   // (the warm streams 8 a batch; DeepSeek-V4 routes 6)
        for (auto& sbuf : F->stage)
            if (cudaHostAlloc((void**) &sbuf, sstride, cudaHostAllocPortable) != cudaSuccess) {
                err = "glm fast: pinned disk staging did not allocate";
                return false;
            }
        for (size_t c = 0; c < cls_stride.size() && wsum > 0; ++c) {
            auto& R = F->rc[c];
            R.stride = cls_stride[c];
            int64_t n = cls_n[c];
            const int64_t cap = cls_cap[c];
            const int64_t least = std::min<int64_t>(16, cap);   // the floor, or the whole cap when that is smaller
            void* p = nullptr;
            // a refusal near a pinning cap: 1/32 less a try for the first four (a tier just over a cap loses ~3%, not
            // the 12% of a 7/8 step), then 1/8 less
            for (int tries = 0; n >= least; ++tries) {
                if ((p = numa_pinned((size_t) n * R.stride)) != nullptr) {   // (over every socket: numa_pinned)
                    R.registered = true;
                    break;
                }
                if (cudaHostAlloc(&p, (size_t) n * R.stride, cudaHostAllocPortable) == cudaSuccess) break;
                cudaGetLastError();
                p = nullptr;
                const int64_t less = tries < 4 ? n * 31 / 32 : n * 7 / 8;
                n = (n > least && less < least) ? least : less;
            }
            if (p == nullptr) {
                err = "glm fast: the pinned RAM tier did not allocate";
                return false;
            }
            // (an APU whose pool holds every expert keeps a staging-only tier: no expert is outside VRAM to hold)
            if (F->ram_resident && !staging_only && n < cap) {
                err = "glm fast: --glm-ram-resident: the RAM tier cannot hold every non-VRAM expert (class " +
                      std::to_string(c) + ": " + std::to_string(n) + " of " + std::to_string(cap) +
                      " slots pinned; raise STRATA_GLM_RAM_GB, its headroom, or lower --max-context)";
                return false;
            }
            R.base = (uint8_t*) p;
            R.n = (int) n;
            R.key.assign((size_t) n, -1);
            R.st.assign((size_t) n, FastState::kRFree);
            R.tick.assign((size_t) n, 0);
            F->ram_bytes += (size_t) n * R.stride;
        }
        if (split_load_ == nullptr && !staging_only && ram_budget_ < 0 && remaining >= 0) {
            remaining = std::max<int64_t>(0, remaining - (int64_t) F->ram_bytes);
            remaining_layers -= n_moe;
        }
        // pinned memory cannot be swapped out: say so when what is left free is short of the headroom (a parallel
        // load: once every part has pinned, ram_left_check)
        if (!staging_only && !fixed_ram && ram_budget_ < 0 && split_load_ != nullptr) {
            F->ram_check_head = head_b;
        } else if (!staging_only && !fixed_ram && ram_budget_ < 0) {
            const int64_t left = avail_ram_now();
            if (left > 0 && left - unseen_pf < head_b / 2)
                std::fprintf(stderr, "glm fast: WARNING - CUDA%d: %.1f GB of RAM free after its RAM tier, and the prompt "
                                     "path still pins up to %.1f GB: if the system swaps or stalls, set "
                                     "STRATA_GLM_RAM_GB lower\n", dev_, (double) left / 1073741824.0,
                             (double) unseen_pf / 1073741824.0);
        }
        int64_t ram_slots = 0;
        for (auto& R : F->rc) ram_slots += R.n;
        std::fprintf(stderr, "glm fast: CUDA%d RAM tier %.2f GB pinned, %lld slots\n", dev_,
                     (double) F->ram_bytes / 1073741824.0, (long long) ram_slots);
        if (F->ram_resident)
            std::fprintf(stderr, "glm fast: CUDA%d RAM-resident mode: every non-VRAM expert gets a slot (+%d slack/layer); disk evictions are OFF\n",
                         dev_, F->ram_slack);
        // the prompt path's lend cap (Strata's, from the share of the expert bytes held pinned), now that it is known
        if (pf_ != nullptr) {
            double held = (double) F->ram_bytes, all = 0;
            for (int il = l0_; il < lt_; ++il)
                if (F->L[(size_t) il].moe) {
                    const auto& P = F->lp[(size_t) il];
                    all += (double) g.n_expert * (double) P.stride;
                    held += (double) std::min(g.n_expert, P.n) * (double) P.stride;
                }
            prefill_settle(all > 0 ? std::min(1.0, held / all) : 1.0);
        }
        F->upd_at.assign((size_t) (2 * F->n_keys + NL * gf::kSpares), 0);
        F->upd_gen.assign(F->upd_at.size(), 0u);
        // disk reads: three slices per expert in parallel, through the staging above when the RAM tier has no free slot
        F->workers.reset(new Workers(8));
        // LOOKAHEAD: the predictions per layer, a load state per RAM slot, and the reader threads
        if (const char* ah = getenv("STRATA_GLM_AHEAD")) F->n_ahead = std::max(0, std::min(gf::kAhead, std::atoi(ah)));
        if (g.n_expert > 512) F->n_ahead = 0;
        const char* ahr = getenv("STRATA_GLM_AHEAD_READ");
        F->ahead_read = F->n_ahead > 0 && (ahr == nullptr || std::atoi(ahr) != 0);
        std::array<std::array<short, 8>, gf::kAhead> none;
        for (auto& r : none) r.fill(-1);
        F->ah_pred.assign((size_t) NL, none);
        for (auto& R : F->rc) {
            F->rload.emplace_back(new std::atomic<uint64_t>[(size_t) R.n]);
            for (int s = 0; s < R.n; ++s) F->rload.back()[(size_t) s].store(0);
        }
        if (F->ahead_read)
            for (int i = 0; i < 2; ++i) F->ah_th.emplace_back([this] { fast_ahead_reader(); });
        const char* wv = getenv("STRATA_GLM_WARM");
        if ((wv == nullptr || std::atoi(wv) != 0) && !fast_warm(err)) return false;
        // RAM-resident mode is all-or-nothing: after the warm, every expert must be in VRAM or the RAM tier -
        // one that fits neither would be read from disk at runtime (STRATA_GLM_WARM=0 fills on demand: no check)
        if (F->ram_resident && (wv == nullptr || std::atoi(wv) != 0)) {
            int uncovered = 0, total = 0;
            for (int il = l0_; il < lt_; ++il) {
                if (!F->L[(size_t) il].moe) continue;
                total += g.n_expert;
                for (int e = 0; e < g.n_expert; ++e) {
                    const size_t key = (size_t) il * g.n_expert + e;
                    uncovered += F->slot_of[key] < 0 && F->ram_of[key] < 0;
                }
            }
            if (uncovered > 0) {
                err = "glm fast: --glm-ram-resident: " + std::to_string(uncovered) +
                      " experts fit neither tier after the warm - raise STRATA_GLM_RAM_GB or lower --max-context";
                return false;
            }
            std::fprintf(stderr, "glm fast: CUDA%d RAM-resident verified: all %d experts are in VRAM or pinned RAM\n",
                         dev_, total);
        }
        F->lane_pending = true;
        if (!defer_lane_ && !fast_setup_finish(err)) return false;
    }
    if (const char* mp = getenv("STRATA_DSV4_MTP"); mp != nullptr && mp[0] && g.dsv4() && d4mtp_ == nullptr) {
        cudaSetDevice(dev_);
        if (!d4_mtp_load(mp, err)) return false;
        // the speculative decode on it unless STRATA_DSV4_MTP_SPEC=0 (then: the shadow measurement) - with the
        // block's experts in VRAM (staged a token, a draft costs more than it saves)
        const char* v = getenv("STRATA_DSV4_MTP_SPEC");
        if ((v == nullptr || std::atoi(v) != 0) && split_next_ == nullptr && d4mtp_resident_ && !d4_spec_alloc(err))
            return false;
    }
    return true;
}

// A parallel split load checks the free RAM once every part has pinned its tier (a part's own check would see the
// others half pinned)
void Glm5Model::ram_left_check() const {
    for (const Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        const FastState* F = m->fast_;
        if (F == nullptr || F->ram_check_head <= 0) continue;
        const int64_t left = avail_ram_now();
        if (left > 0 && left < F->ram_check_head / 2)
            std::fprintf(stderr, "glm fast: WARNING - %.1f GB of RAM free after the split's RAM tiers: if the system "
                                 "swaps or stalls, set STRATA_GLM_RAM_GB lower\n", (double) left / 1073741824.0);
        return;
    }
}

bool Glm5Model::fast_setup_finish(std::string& err) {
    FastState* F = fast_;
    if (F == nullptr || !F->lane_pending) return true;
    F->lane_pending = false;
    cudaSetDevice(dev_);
    if (!fast_cpu_lane_setup(err)) return false;
    F->svc = std::thread([this] { fast_service(); });
    glmfast::pin_thread(F->svc.native_handle(), F->cpu_pin);   // (the CPU lane's last worker: on its pool's node)
    return true;
}

// ---------------------------------------------------------------- the CPU lane
// ne RAM-tier experts of layer il (blob[i]: [gate | up | down] as in the tiers) on the CPU pool: out = sum_i w[i] *
// expert_i(x), in i order.  Gate/up rows then down rows, each split in row chunks across the pool and the caller.
void Glm5Model::fast_cpu_experts(int il, int ne, const uint8_t* const* blob, const float* w, const float* x, float* out) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    const auto& Ly = F->L[(size_t) il];
    const auto& nf = F->cpu_fmt[(size_t) il];
    namespace kc = strata::kernels::cpu;
    const int n_ff = g.n_ff_exp, n_embd = g.n_embd;
    // one job per pool thread: a contiguous run of the call's rows, the experts' gate/up rows end to end (then the
    // down rows likewise), the weighted sum after.  Each thread streams its share in one piece - 2-socket Xeon, 5
    // Q3_K experts from DRAM, 40 threads: 0.154 ms an expert vs 0.201 with 32/64-row jobs claimed in turn (the short
    // streams held the memory system to ~55 of the ~92 GB/s one socket reads; a pure read of the same jobs did too).
    // The run is cut in contiguous pieces claimed in turn, about 48 in all (STRATA_GLM_CPU_SPLIT=<pieces a thread>
    // overrides): with few threads a thread the disk readers or the service thread delay no longer holds up the whole
    // call, with many each still streams its share in one piece.  1x V100, 6-core i5, Maya-S decode: one piece a thread
    // 14.3 tok/s (the lane 34 ms a token), 2: 15.7, 4: 16.8, 8: 17.1 (21 ms; v1.0.11's 64-row jobs 16.8); the 2-socket
    // Xeon above, 40 threads: one piece each
    static const int kSplit = [] {
        const char* v = getenv("STRATA_GLM_CPU_SPLIT");
        return v ? std::max(1, std::min(64, std::atoi(v))) : 0;
    }();
    const int threads = F->cpu_pool->active();
    const int T = threads * (kSplit > 0 ? kSplit : std::max(1, std::min(8, 48 / std::max(1, threads))));
    kc::native_quant_act(nf, x, F->cpu_act.data());
    const int gu_tot = ne * n_ff;
    F->cpu_pool->run(T, [&](int job) {
        const void* a[1] = {F->cpu_act.data()};
        const int q1 = (int) ((int64_t) gu_tot * (job + 1) / T);
        for (int q = (int) ((int64_t) gu_tot * job / T); q < q1;) {
            const int i = q / n_ff, r0 = q % n_ff, r1 = std::min(n_ff, r0 + (q1 - q));
            float* o[1] = {F->cpu_ff.data() + (size_t) i * n_ff};
            kc::native_gu_rows_split(nf, blob[i], blob[i] + Ly.gu_bytes, a, 1, o, r0, r1, g.swiglu_exp);
            q += r1 - r0;
        }
    });
    for (int i = 0; i < ne; ++i)
        kc::native_quant_h(nf, F->cpu_ff.data() + (size_t) i * n_ff, F->cpu_hq.data() + (size_t) i * kc::kNativeHBytes);
    const int dn_tot = ne * n_embd;
    F->cpu_pool->run(T, [&](int job) {
        const int q1 = (int) ((int64_t) dn_tot * (job + 1) / T);
        for (int q = (int) ((int64_t) dn_tot * job / T); q < q1;) {
            const int i = q / n_embd, r0 = q % n_embd, r1 = std::min(n_embd, r0 + (q1 - q));
            const void* h[1] = {F->cpu_hq.data() + (size_t) i * kc::kNativeHBytes};
            float* o[1] = {F->cpu_dn.data() + (size_t) i * n_embd};
            kc::native_down_rows_split(nf, blob[i] + Ly.down_off, h, 1, o, r0, r1);
            q += r1 - r0;
        }
    });
    for (int r = 0; r < n_embd; ++r) {
        float m = 0.0f;
        for (int i = 0; i < ne; ++i) m += w[i] * F->cpu_dn[(size_t) i * n_embd + r];
        out[r] = m;
    }
    // STRATA_GLM_CPU_LANE_VERIFY=1 (debug, slow): the same experts in double precision from ggml's dequantised rows;
    // the relative L2 distance of the lane's sum, accumulated and printed every 16 calls
    static const bool verify = getenv("STRATA_GLM_CPU_LANE_VERIFY") != nullptr;
    if (verify) {
        static double e2 = 0, n2 = 0, worst = 0;
        static int calls = 0;
        const ggml_type_traits* tg = ggml_get_type_traits((ggml_type) Ly.gu_type);
        const ggml_type_traits* td = ggml_get_type_traits((ggml_type) Ly.d_type);
        std::vector<float> row((size_t) std::max(n_embd, n_ff));
        std::vector<double> hr((size_t) n_ff), ref((size_t) n_embd, 0.0);
        const double lim = g.swiglu_exp;
        for (int i = 0; i < ne; ++i) {
            for (int r = 0; r < n_ff; ++r) {
                double gs = 0.0, us = 0.0;
                tg->to_float(blob[i] + (size_t) r * nf.gu_row, row.data(), n_embd);
                for (int c = 0; c < n_embd; ++c) gs += (double) row[(size_t) c] * x[c];
                tg->to_float(blob[i] + Ly.gu_bytes + (size_t) r * nf.gu_row, row.data(), n_embd);
                for (int c = 0; c < n_embd; ++c) us += (double) row[(size_t) c] * x[c];
                gs = std::min(gs, lim);
                us = std::min(std::max(us, -lim), lim);
                hr[(size_t) r] = gs / (1.0 + std::exp(-gs)) * us;
            }
            for (int r = 0; r < n_embd; ++r) {
                double s = 0.0;
                td->to_float(blob[i] + Ly.down_off + (size_t) r * nf.d_row, row.data(), n_ff);
                for (int j = 0; j < n_ff; ++j) s += (double) row[(size_t) j] * hr[(size_t) j];
                ref[(size_t) r] += w[i] * s;
            }
        }
        double a = 0, b = 0;
        for (int r = 0; r < n_embd; ++r) {
            a += (out[r] - ref[(size_t) r]) * (out[r] - ref[(size_t) r]);
            b += ref[(size_t) r] * ref[(size_t) r];
        }
        e2 += a;
        n2 += b;
        worst = std::max(worst, std::sqrt(a / std::max(b, 1e-30)));
        if (++calls % 16 == 0)
            std::fprintf(stderr, "cpu lane verify (%d calls, layer %d): relative L2 error %.5f, worst call %.5f\n", calls, il,
                         std::sqrt(e2 / n2), worst);
    }
}

// The CPU lane's pool a split's parts share (fast_cpu_lane_setup): made by the first part, found by the later ones -
// one model a process; a reload, or another count, makes a new one.
static std::shared_ptr<glmfast::Workers> shared_cpu_pool(int threads) {
    static std::mutex mu;
    static std::weak_ptr<glmfast::Workers> held;
    std::lock_guard<std::mutex> lk(mu);
    std::shared_ptr<glmfast::Workers> p = held.lock();
    if (p == nullptr || p->size() != threads) {
        p = std::make_shared<glmfast::Workers>(threads - 1, 20000);
        held = p;
    }
    return p;
}

// This process's physical cores (distinct package/core pairs among the CPUs it may run on); 0 when unknown.
static int physical_cores() {
#ifdef __linux__
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof set, &set) != 0) return 0;
    std::vector<std::pair<int, int>> seen;
    for (int c = 0; c < CPU_SETSIZE; ++c) {
        if (!CPU_ISSET(c, &set)) continue;
        const std::string b = "/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/";
        int pk = -1, co = -1;
        std::ifstream(b + "physical_package_id") >> pk;
        std::ifstream(b + "core_id") >> co;
        if (co < 0) return 0;
        if (std::find(seen.begin(), seen.end(), std::make_pair(pk, co)) == seen.end()) seen.push_back({pk, co});
    }
    return (int) seen.size();
#elif defined(_WIN32)
    // Windows: the processor cores (each one entry, whatever its SMT threads)
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (len == 0) return 0;
    std::vector<uint8_t> buf(len);
    auto* info = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*) buf.data();
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, info, &len)) return 0;
    int cores = 0;
    for (DWORD at = 0; at < len;) {
        auto* e = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*) (buf.data() + at);
        if (e->Relationship == RelationProcessorCore) ++cores;
        at += e->Size;
    }
    return cores;
#else
    return 0;
#endif
}

// STRATA_GLM_CPU_CAL file: replace key's line with vals (empty: drop it)
static void cal_put(const std::string& file, const std::string& key, const std::string& vals) {
    std::string keep;
    if (FILE* cf = std::fopen(file.c_str(), "r")) {
        char line[2048];
        while (std::fgets(line, sizeof line, cf)) {
            const char* tab = std::strrchr(line, '\t');
            if (tab == nullptr || std::string(line, (size_t) (tab - line)) != key) keep += line;
        }
        std::fclose(cf);
    }
    if (!vals.empty()) keep += key + "\t" + vals + "\n";
    const std::string tmp = file + ".tmp";
    if (FILE* cf = std::fopen(tmp.c_str(), "w")) {
        const bool ok = std::fputs(keep.c_str(), cf) >= 0;
        if (std::fclose(cf) == 0 && ok) {
            std::error_code ec;
            std::filesystem::rename(tmp, file, ec);
        }
    }
}

static std::string cal_vals(double c, double p, double ps, double ref) {
    char v[128];
    std::snprintf(v, sizeof v, ref > 0.0 ? "%.4f %.4f %.4f %.4f" : "%.4f %.4f %.4f", c, p, ps, ref);
    return v;
}

// The CPU LANE, on by default: one thread per physical core (split evenly across the parts of a layer split);
// STRATA_GLM_CPU_LANE=<threads> sets the count, 0 turns it off; STRATA_GLM_CPU_LANE<n>=<threads> sets one GPU's own
// (CUDA<n>: a slow link wants more threads than a fast one).  Measures this machine once - one expert on the CPU
// pool vs one over PCIe - and derives the split: of f RAM-tier experts in a route, the k the host computes so that
// the slower of the two lanes finishes first (STRATA_GLM_CPU_PLAN=<digits for f = 0..8> overrides).  A CPU slower
// than the PCIe link at every f leaves the lane off.
bool Glm5Model::fast_cpu_lane_setup(std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    const char* lv = getenv(("STRATA_GLM_CPU_LANE" + std::to_string(dev_)).c_str());
    const bool own_count = lv != nullptr;   // this card's own count: a pool of its own
    if (lv == nullptr) lv = getenv("STRATA_GLM_CPU_LANE");
    // a split of 3+ GPUs runs its parts one after another - a decode token visits the cards in turn, and the pipelined
    // speculative decode, which runs two at once, is a two-part one - so ONE pool serves every part, sized as one
    // GPU's (Strata: one expert pool for the process).  A pool per part, cores / parts each, left most of the CPU idle -
    // and as idle workers spin 20 ms after a batch, more threads a part would have spun against the card at work.
    // Maya-L on 9 GPUs, 88 vCPUs, fresh prompts (95.8% VRAM hits): 9 threads a card 0.74 ms a RAM-tier expert, 25.7
    // tok/s; one pool of 40 0.32 ms, 30.7 tok/s; an 8K prompt, whose chunks take the pool in turn, 1265 -> 1466 tok/s.
    // STRATA_GLM_CPU_SHARED=0: a pool per part (A/B); 1: one pool for a two-GPU split too.
    const char* shv = getenv("STRATA_GLM_CPU_SHARED");
    const bool shared = n_parts_ >= 2 && !own_count &&
                        (shv != nullptr && shv[0] != '\0' ? std::atoi(shv) != 0 : n_parts_ >= 3);
    int threads = 0;
    if (lv != nullptr) {
        threads = std::max(0, std::min(64, std::atoi(lv)));
    } else {
        int cores = physical_cores();
        if (cores <= 0) cores = (int) std::thread::hardware_concurrency() / 2;
        threads = cores;
        // cores without SMT siblings (Intel's Arrow Lake, E-core parts): a thread a core spins on EVERY CPU, and the
        // disk readers, the service and the main thread wait for a time slice - two thirds of them leave the rest free
        // (24-core Ultra 9 275HX, Windows, 33 GB RAM tier, the same 36.6 disk reads a token: 24 threads 2.1 tok/s and
        // a disk wait 16.8 ms, 20 3.5 tok/s, 16 3.8 tok/s and 8.1 ms, 12 3.8 tok/s; shard views released: 24 threads
        // 5.7 tok/s, 16 7.6)
        if (const int logical = (int) std::thread::hardware_concurrency(); logical > 0 && cores >= logical)
            threads = logical * 2 / 3;
        // one GPU: at most one NUMA node's worth, less 4 - a spinning pool on both sockets syncs across them, and the
        // CPUs left free keep the main thread's event wait, the service and warm-up threads off the spinning ones
        // (2-socket Xeon, 88 vCPUs, Q3_K experts: 88 threads 531 us an expert, 44 216 us, 44 with the RAM tier
        // interleaved 152 us - ~85 of the ~105 GB/s this host reads; a pool per socket, tried, did not add to that)
        if (n_parts_ == 1 || shared)
            if (const int node = numa_node_cpus(); node > 0 && threads > node - 4) threads = node - 4;
        if (shared)
            threads = std::min(threads, cores - (n_parts_ - 1));   // a core for each other part's service thread
        else
            threads /= std::max(1, n_parts_);   // a two-part split's parts can run at once: one pool each
        if (threads < 2) threads = 0;   // one core is the service thread's
    }
    // a split whose parts have a pool each (two GPUs, or STRATA_GLM_CPU_SHARED=0): each part's pool on CPUs of its own
    // (part_cpus: whole NUMA nodes while there are enough, runs of whole cores of a shared one otherwise), at most those
    // CPUs less the part's share of the 4 a node keeps free, like one GPU's - the shared pool is one GPU's, unpinned.  Unpinned, the parts' pools (half the vCPUs each) shared the cores: a part works while the other's idle
    // workers spin, and the 2-socket Xeon's 3090 + 5060 Ti split took 0.28 ms an expert where one GPU's pool took 0.15
    // (and with MTP drafting both parts run at once).  The RAM tier stays interleaved (one socket reading only its own
    // memory: 54 GB/s, interleaved 77).  By default only on a host with two NUMA nodes or more, where it measured
    // faster; on one node the unpinned pools sharing every core were (two V100s, one 14-core Xeon: 25.8 tok/s against
    // 24.7 pinned).  STRATA_GLM_CPU_PIN=1 / 0 pins / unpins on any host.
    std::vector<int> pin_cpus;
    int pin_node = -1;
    const char* pin_env = getenv("STRATA_GLM_CPU_PIN");
    const bool pin = pin_env != nullptr && pin_env[0] ? std::atoi(pin_env) != 0 : numa_nodes().size() >= 2;
    if (n_parts_ >= 2 && !shared && threads > 0 && pin) {
        std::vector<int> gpu_node;
        for (int d : split_devs_) gpu_node.push_back(gpu_numa_node(d));
        int spare = 4;
        pin_cpus = part_cpus(part_, n_parts_, gpu_node, pin_node, spare);
        // a thread a physical core, as unpinned - their SMT siblings are the CPUs left free (one CCD of a 5900X a
        // part: 10 threads on its 6 cores 10.88 tok/s, 6 threads 11.45, #77); no siblings: those CPUs less the spare
        if (!pin_cpus.empty() && lv == nullptr) {
            std::vector<std::pair<int, int>> pc;
            for (int c : pin_cpus) pc.push_back(cpu_core(c));
            std::sort(pc.begin(), pc.end());
            const int n_cores = (int) (std::unique(pc.begin(), pc.end()) - pc.begin());
            threads = n_cores < (int) pin_cpus.size() ? std::max(2, n_cores)
                                                      : std::max(2, (int) pin_cpus.size() - spare);
        }
    }
    if (threads <= 0 || g.n_embd > 4096 || g.n_exp_used > 8) return true;
    namespace kc = strata::kernels::cpu;
    F->cpu_fmt.assign(F->L.size(), kc::NativeFmt{});
    int il_cal = -1;
    for (int il = l0_; il < lt_; ++il) {
        const auto& Ly = F->L[(size_t) il];
        if (!Ly.moe || Ly.mtp) continue;
        std::string e2;
        kc::NativeFmt f;
        if (kc::native_fmt(Ly.gu_type, Ly.d_type, g.n_embd, g.n_ff_exp, f, e2) && f.h_bytes <= kc::kNativeHBytes)
            F->cpu_fmt[(size_t) il] = f;
        if (il_cal < 0 && F->cpu_fmt[(size_t) il].n_ff > 0) il_cal = il;
    }
    if (il_cal < 0) return true;
    if (const char* ck = getenv("STRATA_GLM_CPU_LANE_CHECK"))   // the check's layer (below)
        if (std::atoi(ck) > 0 && std::atoi(ck) < (int) F->cpu_fmt.size() && F->cpu_fmt[(size_t) std::atoi(ck)].n_ff > 0)
            il_cal = std::atoi(ck);
    // a calibration blob: a RAM-tier expert of that layer (the warm-up filled the tier; a class holds every layer of
    // its blob size)
    const auto& R = F->rc[(size_t) F->layer_rc[(size_t) il_cal]];
    // (up to 4 of them: a route's RAM-tier experts come several to a call, and one alone leaves most of the pool idle -
    // timed alone, a 40-thread pool's expert took 0.26 ms where a decode's calls ran them at 0.16, and the split
    // sent the 8th of 8 over PCIe, 2.1 ms on the critical path)
    // Each call takes the next 4 of up to 64 (~480 MB of Maya-S24's 7.5 MB experts): a decode reads its RAM-tier
    // experts from DRAM, and 4 timed again and again stay in the CPU's L3 - a 7950X (64 MB L3) timed 0.097 ms an
    // expert where decode ran them at ~0.16 (DRAM-bound, 47 of its 57 GB/s), and the split kept 4 and 5 of a layer's
    // RAM-tier experts on the CPU: 25.1 / 25.3 tok/s against 26.9 / 27.5 with one of them over PCIe
    constexpr int kCalSets = 16;
    std::vector<const uint8_t*> cal_all;
    for (int s = 0; s < R.n && (int) cal_all.size() < 4 * kCalSets; ++s)
        if (R.st[(size_t) s] == FastState::kRHold && R.key[(size_t) s] / g.n_expert == il_cal)
            cal_all.push_back(R.base + (size_t) s * R.stride);
    if (cal_all.empty()) return true;   // nothing in RAM: the lane would never run
    const uint8_t* cal = cal_all[0];
    const int n_cal = (int) std::min<size_t>(4, cal_all.size());
    const int n_sets = (int) cal_all.size() / n_cal;   // (fewer than 64 held: the sets repeat sooner)
    auto cal_set = [&](int i) { return cal_all.data() + (size_t) (i % n_sets) * n_cal; };
    void* hp = nullptr;
    if (cudaHostAlloc(&hp, sizeof(gf::CpuAnswer), cudaHostAllocMapped) != cudaSuccess ||
        cudaMalloc((void**) &F->cpu_seq_d, 64) != cudaSuccess) {
        err = "glm fast: the CPU lane's buffers did not allocate";
        return false;
    }
    std::memset(hp, 0, sizeof(gf::CpuAnswer));
    cudaMemset(F->cpu_seq_d, 0, 64);
    F->cpu_ans_h = (gf::CpuAnswer*) hp;
    void* dp = nullptr;
    cudaHostGetDevicePointer(&dp, hp, 0);
    F->cpu_act.assign(kc::kNativeActBytes, 0);
    F->cpu_hq.assign((size_t) 8 * kc::kNativeHBytes, 0);
    F->cpu_ff.assign((size_t) 8 * g.n_ff_exp, 0.0f);
    F->cpu_dn.assign((size_t) 8 * g.n_embd, 0.0f);
    // the service thread is the pool's last worker; idle workers spin 20 ms (a decode's routes come ~1 ms apart)
    if (shared) {
        F->cpu_pool = shared_cpu_pool(threads);
    } else {
        F->cpu_pool.reset(new glmfast::Workers(threads - 1, 20000, pin_cpus));
        F->cpu_node = pin_node;
        F->cpu_pin = pin_cpus;
    }
    // STRATA_GLM_CPU_CAL=<file>: reuse the timed values across starts, keyed by build, model, layer, card and pool
    const char* calf = getenv("STRATA_GLM_CPU_CAL");
    std::string cal_key;
    double c_ms = 0.0, p_ms = 0.0, ps_ms = 0.0;
    bool cached = false;
    if (calf != nullptr && *calf != '\0') {
        char bus[32] = "?";
        if (cudaDeviceGetPCIBusId(bus, (int) sizeof bus, dev_) != cudaSuccess) cudaGetLastError();
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, dev_) != cudaSuccess) {
            cudaGetLastError();
            prop.name[0] = 0;
        }
        std::error_code ec;
        const std::filesystem::path pd = std::filesystem::weakly_canonical(pack_dir_, ec);
        cal_key = std::string(STRATA_VERSION) + " " + __DATE__ + " " + __TIME__ + " | " + (ec ? pack_dir_ : pd.string()) + " | layer " + std::to_string(il_cal) +
                  " | " + std::to_string(F->L[(size_t) il_cal].blob) + " B | " + bus + " " + prop.name + " | " +
                  std::to_string(threads) + (shared ? " threads shared" : " threads") + " | " + std::to_string(n_cal);
        if (FILE* cf = std::fopen(calf, "r")) {
            char line[2048];
            while (!cached && std::fgets(line, sizeof line, cf)) {
                char* tab = std::strrchr(line, '\t');
                if (tab == nullptr) continue;
                *tab = 0;
                double ref = 0.0;
                cached = cal_key == line && std::sscanf(tab + 1, "%lf %lf %lf %lf", &c_ms, &p_ms, &ps_ms, &ref) >= 3 &&
                         c_ms > 0.0 && p_ms > 0.0 && ps_ms > 0.0;
                if (cached) F->cal_ref = ref;
            }
            std::fclose(cf);
        }
    }
    const size_t blob = F->L[(size_t) il_cal].blob;
    if (!cached) {
    // ---- calibration, each lane alone: the CPU after 100 ms of the same work (an idle CPU's clocks take tens of ms to
    //      ramp up - a decode keeps them up), then five rounds of 16 runs ~100 ms apart (the same work between them
    //      keeps the clocks up), and an expert's time is the fastest round's median.  One round takes ~40 ms, so a
    //      burst of other work on the CPU (the OS reclaiming memory after the RAM tier was pinned, a desktop's
    //      background jobs) covered all of its runs at once: one start timed 0.63 ms an expert against the usual 0.32,
    //      its CPU lane took half its share and decode lost 16% for the engine's life (#56).  Interference only adds
    //      time, so the fastest round is the CPU's own speed
    const auto median16 = [](std::array<double, 16>& v) {
        std::sort(v.begin(), v.end());
        return 0.5 * (v[7] + v[8]);
    };
    std::vector<float> x((size_t) g.n_embd), out((size_t) g.n_embd);
    for (int i = 0; i < g.n_embd; ++i) x[(size_t) i] = 0.01f * (float) ((i * 37) % 101 - 50);
    const float w4[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    int set = 0;
    const auto busy = [&](int ms) {
        for (const auto tw = std::chrono::steady_clock::now();
             std::chrono::steady_clock::now() - tw < std::chrono::milliseconds(ms);)
            fast_cpu_experts(il_cal, n_cal, cal_set(set++), w4, x.data(), out.data());
    };
    busy(100);
    std::array<double, 16> runs{};
    double best = 0.0;
    for (int round = 0; round < 5; ++round) {
        if (round > 0) busy(100);
        for (int rep = 0; rep < 16; ++rep) {
            const uint8_t* const* cals = cal_set(set++);
            const auto t0 = std::chrono::steady_clock::now();
            fast_cpu_experts(il_cal, n_cal, cals, w4, x.data(), out.data());
            runs[(size_t) rep] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        }
        const double med = median16(runs);
        best = round == 0 ? med : std::min(best, med);
    }
    c_ms = best / n_cal;   // an expert's share of a call
    cudaEvent_t e0 = nullptr, e1 = nullptr;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    // the link is timed with the GPU awake (see glm_link_wake): 400 ms for the clocks and the link to ramp up, then
    // 1.1 s of the kernel left for both measurements below
    cudaStream_t wake = nullptr;
    if (cudaStreamCreateWithFlags(&wake, cudaStreamNonBlocking) == cudaSuccess) {
        glm_link_wake<<<1, 1, 0, wake>>>(1500ull * 1000000ull);
        std::this_thread::sleep_for(std::chrono::milliseconds(400));
    }
    for (int rep = 0; rep < 24; ++rep) {   // (8 to warm up, then 16 timed: their median, as the CPU's)
        cudaEventRecord(e0, F->cs);
        cudaMemcpyAsync(F->scratch, cal, blob, cudaMemcpyHostToDevice, F->cs);
        cudaEventRecord(e1, F->cs);
        cudaEventSynchronize(e1);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, e0, e1);
        if (rep >= 8) runs[(size_t) (rep - 8)] = ms;
    }
    p_ms = median16(runs);
    // ... and its streaming rate - copies back to back, the way the prompt path stages experts (a 3090 on PCIe 3.0 x8:
    // 2.09 ms for one copy, 1.57 a copy in a stream): the prompt's CPU / PCIe split plans with this one
    ps_ms = p_ms;
    {
        cudaEventRecord(e0, F->cs);
        for (int rep = 0; rep < 8; ++rep) cudaMemcpyAsync(F->scratch, cal, blob, cudaMemcpyHostToDevice, F->cs);
        cudaEventRecord(e1, F->cs);
        cudaEventSynchronize(e1);
        float ms = 0.0f;
        if (cudaEventElapsedTime(&ms, e0, e1) == cudaSuccess && ms > 0.0f) ps_ms = std::min(p_ms, (double) ms / 8.0);
    }
    if (wake != nullptr) {
        cudaStreamSynchronize(wake);
        cudaStreamDestroy(wake);
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    if (!cal_key.empty()) cal_put(calf, cal_key, cal_vals(c_ms, p_ms, ps_ms, 0.0));
    }
    F->cpu_c_ms = c_ms;   // (the prompt path splits its staged experts by them too)
    F->cpu_p_ms = p_ms;
    F->cpu_ps_ms = ps_ms;
    // the split: k of f to the host minimises max(PCIe (f - k) p, lane overhead + k c); ties keep the PCIe lane
    const double over_ms = 0.04;
    unsigned long long plan = 0;
    std::string tab;
    const char* pv = getenv("STRATA_GLM_CPU_PLAN");
    // STRATA_GLM_PCIE_SHARE=<0..1> (setup's calibration): the share of a route's RAM-tier experts that go over PCIe
    const char* sv = getenv("STRATA_GLM_PCIE_SHARE");
    const double share = sv != nullptr ? std::max(0.0, std::min(1.0, std::atof(sv))) : -1.0;
    for (int f = 0; f <= 8; ++f) {
        int k = 0;
        if (pv != nullptr) {
            k = f < (int) std::strlen(pv) ? std::max(0, std::min(f, pv[f] - '0')) : 0;
        } else if (share >= 0.0) {
            k = f - (int) std::lround(share * f);
        } else {
            double best = p_ms * f;
            for (int kk = 1; kk <= f; ++kk) {
                const double t = std::max(p_ms * (f - kk), over_ms + c_ms * kk);
                if (t < best - 1e-9) {
                    best = t;
                    k = kk;
                }
            }
        }
        plan |= (unsigned long long) k << (4 * f);
        tab += " " + std::to_string(k);
    }
    if (plan == 0ull) {
        std::fprintf(stderr, "glm fast: CUDA%d CPU lane off: an expert %.3f ms on the CPU vs %.3f ms over PCIe\n", dev_,
                     c_ms, p_ms);
        F->cpu_pool.reset();
        return true;
    }
    F->cpu_plan = plan;
    F->cpu_plan_start = plan;
    if (!cal_key.empty()) F->cal_file = calf, F->cal_key = cal_key;
    F->md.cpu_seq = F->cpu_seq_d;
    F->md.cpu_ans = dp;
    // the route counts that pick the coldest RAM-tier experts for the host, seeded with the tiers' LFU counts
    // (STRATA_GLM_CPU_COLD=0: the last ones in route order instead)
    const char* cv = getenv("STRATA_GLM_CPU_COLD");
    if ((cv == nullptr || std::atoi(cv) != 0) && F->cnt.size() == (size_t) F->n_keys) {
        if (cudaMalloc((void**) &F->dcnt_d, (size_t) F->n_keys * sizeof(unsigned int)) != cudaSuccess) {
            err = "glm fast: the CPU lane's route counts did not allocate";
            return false;
        }
        cudaMemcpy(F->dcnt_d, F->cnt.data(), (size_t) F->n_keys * sizeof(unsigned int), cudaMemcpyHostToDevice);
        F->md.dcnt = F->dcnt_d;
    }
    std::fprintf(stderr, "glm fast: CUDA%d CPU lane: %d threads%s, an expert %.3f ms on the CPU vs %.3f ms over PCIe -> "
                         "of 0..8 RAM-tier experts the CPU takes%s\n", dev_, threads,
                 shared ? " (one pool, shared by the split's GPUs)"
                 : pin_cpus.empty() ? ""
                 : (" on " + std::to_string(pin_cpus.size()) + " CPUs of its own" +
                    (pin_node >= 0 ? " (NUMA node " + std::to_string(pin_node) + ")" : std::string())).c_str(),
                 c_ms, p_ms, tab.c_str());
    if (cached) std::fprintf(stderr, "glm fast: CUDA%d CPU lane: those times from %s (not timed)\n", dev_, calf);
    // STRATA_GLM_CPU_LANE_CHECK=1: the calibration expert on one normalised input three ways - the device's decode
    // kernels, the CPU lane, and a double-precision reference from ggml's dequantised rows - and their distances
    if (getenv("STRATA_GLM_CPU_LANE_CHECK") != nullptr) {
        const int E = g.n_embd, FFE = g.n_ff_exp;
        const auto& Ly = F->L[(size_t) il_cal];
        const auto& nf = F->cpu_fmt[(size_t) il_cal];
        float *dh = nullptr, *dw = nullptr, *dx = nullptr, *dout = nullptr;
        void *dxq = nullptr, *dhq = nullptr;
        cudaMalloc(&dh, (size_t) E * 4);
        cudaMalloc(&dw, (size_t) E * 4);
        cudaMalloc(&dx, (size_t) E * 4);
        cudaMalloc(&dout, (size_t) E * 4);
        cudaMalloc(&dxq, (size_t) E / 32 * 36);
        cudaMalloc(&dhq, (size_t) 8 * FFE / 32 * 36);
        std::vector<float> h((size_t) E), ones((size_t) E, 1.0f), xc((size_t) E), og((size_t) E), oc((size_t) E);
        for (int i = 0; i < E; ++i) h[(size_t) i] = std::sin(0.71f * i) + ((i % 97) == 0 ? 6.0f : 0.0f);
        cudaMemcpy(dh, h.data(), (size_t) E * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dw, ones.data(), (size_t) E * 4, cudaMemcpyHostToDevice);
        gf::rms_q8(dh, nullptr, dw, g.norm_eps, E, dx, dxq, F->cs);
        cudaMemcpyAsync(F->scratch, cal, blob, cudaMemcpyHostToDevice, F->cs);
        const unsigned long long sp = (unsigned long long) F->scratch;
        const float one = 1.0f;
        cudaMemcpyAsync(F->md.plan_ptr, &sp, 8, cudaMemcpyHostToDevice, F->cs);
        cudaMemcpyAsync(F->md.plan_w, &one, 4, cudaMemcpyHostToDevice, F->cs);
        cudaMemsetAsync(F->md.cpu_flag, 0, 4, F->cs);
        gf::moe_gate_up(Ly.gu_type, F->md, 1, E, FFE, g.swiglu_exp, dxq, dhq, nullptr, 0, nullptr, 0, nullptr, F->cs);
        gf::moe_down(Ly.d_type, F->md, 1, E, FFE, Ly.down_off, dhq, nullptr, dout, F->cs);
        cudaMemcpyAsync(xc.data(), dx, (size_t) E * 4, cudaMemcpyDeviceToHost, F->cs);
        cudaMemcpyAsync(og.data(), dout, (size_t) E * 4, cudaMemcpyDeviceToHost, F->cs);
        cudaStreamSynchronize(F->cs);
        fast_cpu_experts(il_cal, 1, &cal, &one, xc.data(), oc.data());
        // the reference: rows dequantised by ggml, dots in double
        const ggml_type_traits* tg = ggml_get_type_traits((ggml_type) Ly.gu_type);
        const ggml_type_traits* td = ggml_get_type_traits((ggml_type) Ly.d_type);
        std::vector<float> row((size_t) std::max(E, FFE));
        std::vector<double> hr((size_t) FFE), orf((size_t) E);
        for (int r = 0; r < FFE; ++r) {
            double gs = 0.0, us = 0.0;
            tg->to_float(cal + (size_t) r * nf.gu_row, row.data(), E);
            for (int i = 0; i < E; ++i) gs += (double) row[(size_t) i] * xc[(size_t) i];
            tg->to_float(cal + Ly.gu_bytes + (size_t) r * nf.gu_row, row.data(), E);
            for (int i = 0; i < E; ++i) us += (double) row[(size_t) i] * xc[(size_t) i];
            const double lim = g.swiglu_exp;
            gs = std::min(gs, lim);
            us = std::min(std::max(us, -lim), lim);
            hr[(size_t) r] = gs / (1.0 + std::exp(-gs)) * us;
        }
        for (int r = 0; r < E; ++r) {
            double s = 0.0;
            td->to_float(cal + Ly.down_off + (size_t) r * nf.d_row, row.data(), FFE);
            for (int j = 0; j < FFE; ++j) s += (double) row[(size_t) j] * hr[(size_t) j];
            orf[(size_t) r] = s;
        }
        double nr = 0, eg = 0, ec = 0, egc = 0;
        for (int r = 0; r < E; ++r) {
            nr += orf[(size_t) r] * orf[(size_t) r];
            eg += (og[(size_t) r] - orf[(size_t) r]) * (og[(size_t) r] - orf[(size_t) r]);
            ec += (oc[(size_t) r] - orf[(size_t) r]) * (oc[(size_t) r] - orf[(size_t) r]);
            egc += (double) (og[(size_t) r] - oc[(size_t) r]) * (og[(size_t) r] - oc[(size_t) r]);
        }
        std::fprintf(stderr, "glm fast: CPU lane check (layer %d, types %d/%d): relative L2 error vs the double "
                             "reference: device %.5f, CPU lane %.5f; device vs CPU lane %.5f\n", il_cal, Ly.gu_type,
                     Ly.d_type, std::sqrt(eg / nr), std::sqrt(ec / nr), std::sqrt(egc / nr));
        cudaFree(dh);
        cudaFree(dw);
        cudaFree(dx);
        cudaFree(dout);
        cudaFree(dxq);
        cudaFree(dhq);
    }
    return true;
}

// ---------------------------------------------------------------- the usage profile
// Where this machine's expert usage is kept between sessions: STRATA_GLM_USAGE=<file> (0: none), else
// expert_usage.txt in the pack's folder.
std::string Glm5Model::usage_path() const {
    const char* u = getenv("STRATA_GLM_USAGE");
    if (u != nullptr) return std::string(u) == "0" ? std::string() : std::string(u);
    return pack_dir_.empty() ? std::string() : pack_dir_ + "/expert_usage.txt";
}

// Every half's long-memory routes ("layer e:count ..."), one file, written whole and renamed into place.
bool Glm5Model::save_usage() {
    const std::string up = usage_path();
    if (up.empty() || fast_ == nullptr) return false;
    std::string out = "# strata expert usage: routes per expert across sessions (the warm-up fills the tiers in this "
                      "order)\n";
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        FastState* F = m->fast_;
        if (F == nullptr || F->usage.empty()) continue;
        std::vector<uint32_t> u;
        {
            std::lock_guard<std::mutex> lk(F->mu);
            u = F->usage;
        }
        const int NE = m->g_.n_expert;
        for (int il = m->l0_; il < m->lt_; ++il) {
            std::string line = std::to_string(il);
            bool any = false;
            for (int e = 0; e < NE; ++e) {
                const uint32_t n = u[(size_t) il * NE + e];
                if (n == 0) continue;
                line += " " + std::to_string(e) + ":" + std::to_string(n);
                any = true;
            }
            if (any) out += line + "\n";
        }
    }
    const std::string tmp = up + ".tmp";
    {
        std::ofstream f(tmp, std::ios::trunc);
        if (!f) return false;
        f << out;
        if (!f) return false;
    }
    return std::rename(tmp.c_str(), up.c_str()) == 0;
}

// ---------------------------------------------------------------- load-time warm-up
// VRAM and the pinned RAM tier together hold (nearly) every routed expert, so the whole model is streamed from the
// shards ONCE at load: each layer's most frequently routed experts (the pack's expert_prior.txt, written by
// tools/glm_expert_prior.py from routing traces; id order without it) fill its VRAM slots - all but the spares - and
// the rest fill the RAM tier (a few slots per class stay free for disk reads).  The first request then runs warm and
// the disk is touched again only for experts that fit nowhere.  STRATA_GLM_WARM=0 skips it (the tiers fill on demand).
bool Glm5Model::fast_warm(std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    const auto t0 = std::chrono::steady_clock::now();
    const int NL = g.n_layers + 1;
    std::vector<std::vector<int>> order((size_t) NL);
    {
        std::ifstream pf(pack_dir_ + "/expert_prior.txt");
        std::string line;
        while (std::getline(pf, line)) {
            std::istringstream ss(line);
            int il = -1, e = 0;
            ss >> il;
            if (il < 0 || il >= NL) continue;
            while (ss >> e)
                if (e >= 0 && e < g.n_expert) order[(size_t) il].push_back(e);
        }
    }
    // THIS user's experts first: the usage profile of earlier sessions (save_usage) leads each layer's order, its
    // counts become the long memory again, and the LFU counts start from them scaled to at most 32 (enough to keep
    // the profile's experts against one-off routes, little enough that a new task takes over within ~100 tokens;
    // a restart then hits ~75% instead of ~67% over the first 50 tokens on one V100, simulated)
    // the routing profile's counts per expert id (expert_counts.txt), for the blend below
    std::vector<std::vector<double>> pcount((size_t) NL);
    {
        std::ifstream cf(pack_dir_ + "/expert_counts.txt");
        std::string line;
        while (std::getline(cf, line)) {
            std::istringstream ss(line);
            int il = -1;
            double c = 0;
            ss >> il;
            if (il < 0 || il >= NL) continue;
            while (ss >> c) pcount[(size_t) il].push_back(c);
        }
    }
    static const double profile_w = [] {
        const char* v = getenv("STRATA_GLM_PROFILE_WEIGHT");
        return v != nullptr ? std::max(0.0, std::atof(v)) : 1.0;
    }();
    int n_blend = 0;
    const std::string up = usage_path();
    int n_prof = 0;
    if (!up.empty()) {
        std::ifstream uf(up);
        std::string line;
        while (std::getline(uf, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream ss(line);
            int il = -1;
            ss >> il;
            if (il < l0_ || il >= lt_ || !F->L[(size_t) il].moe) continue;
            std::vector<std::pair<uint32_t, int>> v;
            std::string tok;
            while (ss >> tok) {
                const size_t c = tok.find(':');
                if (c == std::string::npos) continue;
                const int e = std::atoi(tok.substr(0, c).c_str());
                const uint32_t n = (uint32_t) std::strtoul(tok.substr(c + 1).c_str(), nullptr, 10);
                if (e >= 0 && e < g.n_expert && n > 0) v.emplace_back(n, e);
            }
            if (v.empty()) continue;
            std::sort(v.rbegin(), v.rend());
            std::vector<int> o;
            for (auto& p : v) {
                o.push_back(p.second);
                const size_t key = (size_t) il * g.n_expert + p.second;
                F->usage[key] = p.first;
                F->cnt[key] = (uint32_t) std::max<uint64_t>(1, (uint64_t) 32 * p.first / v.front().first);
            }
            // with the pack's routing profile (expert_counts.txt, tools/glm_expert_prior.py over a broad corpus): the
            // order is the blend of the two shares - this machine's history, and the profile, which corrects what the
            // history over-represents (one task, a benchmark) and ranks what it never routed
            if (!pcount[(size_t) il].empty() && profile_w > 0.0) {
                double ut = 0.0, pt = 0.0;
                for (auto& p : v) ut += p.first;
                for (double c : pcount[(size_t) il]) pt += c;
                if (ut > 0.0 && pt > 0.0) {
                    std::vector<double> score((size_t) g.n_expert, 0.0);
                    for (auto& p : v) score[(size_t) p.second] += p.first / ut;
                    for (size_t e = 0; e < pcount[(size_t) il].size() && e < score.size(); ++e)
                        score[e] += profile_w * pcount[(size_t) il][e] / pt;
                    o.resize((size_t) g.n_expert);
                    std::iota(o.begin(), o.end(), 0);
                    std::stable_sort(o.begin(), o.end(), [&](int a, int b) { return score[(size_t) a] > score[(size_t) b]; });
                    ++n_blend;
                }
            }
            o.insert(o.end(), order[(size_t) il].begin(), order[(size_t) il].end());
            order[(size_t) il].swap(o);
            ++n_prof;
        }
        if (n_prof > 0)
            std::fprintf(stderr, "glm fast: CUDA%d the tiers follow this machine's usage profile (%s, %d layers%s)\n", dev_,
                         up.c_str(), n_prof, n_blend > 0 ? ", blended with the pack's routing profile" : "");
    }
    for (int il = 0; il < NL; ++il) {
        auto& o = order[(size_t) il];
        std::vector<char> seen((size_t) g.n_expert, 0);
        std::vector<int> clean;
        for (int e : o)
            if (!seen[(size_t) e]) {
                seen[(size_t) e] = 1;
                clean.push_back(e);
            }
        for (int e = 0; e < g.n_expert; ++e)
            if (!seen[(size_t) e]) clean.push_back(e);
        o.swap(clean);
    }
    struct Job {
        int il, e;
        uint8_t* dst;    // the RAM slot, or the VRAM slot (staged)
        bool vram;
    };
    std::vector<Job> jobs;
    const size_t N = (size_t) F->n_keys;
    std::vector<unsigned long long> th(2 * N + (size_t) NL * gf::kSpares, 0ull);
    std::vector<int> next((size_t) NL, 0);
    for (int il = l0_; il < lt_; ++il) {
        if (!F->L[(size_t) il].moe) continue;
        auto& P = F->lp[(size_t) il];
        const int nv = glmfast::warm_pool_slots(F->unified_memory, P.n, g.n_expert, gf::kSpares);
        for (int j = 0; j < nv; ++j) {
            const int e = order[(size_t) il][(size_t) j];
            const int key = il * g.n_expert + e;
            P.key[(size_t) j] = e;
            P.st[(size_t) j] = FastState::kResident;
            F->slot_of[(size_t) key] = j;
            th[(size_t) key] = (unsigned long long) P.slot_ptr(j);
            jobs.push_back(Job{il, e, P.slot_ptr(j), true});
        }
        next[(size_t) il] = nv;
    }
    // the RAM tier, round-robin over the layers so every layer gets its share of a class that cannot hold all
    std::vector<int> rfree(F->rc.size(), 0), rcur(F->rc.size(), 0);
    for (size_t c = 0; c < F->rc.size(); ++c) rfree[c] = F->rc[c].n;
    for (bool progress = true; progress;) {
        progress = false;
        for (int il = l0_; il < lt_; ++il) {
            if (!F->L[(size_t) il].moe || next[(size_t) il] >= g.n_expert) continue;
            const int c = F->layer_rc[(size_t) il];
            auto& R = F->rc[(size_t) c];
            if (rfree[(size_t) c] <= (F->ram_resident ? 0 : 4)) continue;   // resident: no disk-read slots to keep
            while (rcur[(size_t) c] < R.n && R.st[(size_t) rcur[(size_t) c]] != FastState::kRFree) ++rcur[(size_t) c];
            if (rcur[(size_t) c] >= R.n) continue;
            const int s = rcur[(size_t) c]++;
            const int e = order[(size_t) il][(size_t) next[(size_t) il]++];
            const int key = il * g.n_expert + e;
            R.key[(size_t) s] = key;
            R.st[(size_t) s] = FastState::kRHold;
            F->ram_of[(size_t) key] = s;
            th[N + (size_t) key] = (unsigned long long) (R.base + (size_t) s * R.stride);
            jobs.push_back(Job{il, e, R.base + (size_t) s * R.stride, false});
            --rfree[(size_t) c];
            progress = true;
        }
    }
    // stream them: 8 experts per batch, 3 slices each in parallel; VRAM ones through the pinned staging
    size_t done_bytes = 0, total_bytes = 0;
    for (const auto& j : jobs) total_bytes += F->L[(size_t) j.il].blob;
    int last_pct = -1;
    for (size_t b = 0; b < jobs.size(); b += 8) {
        const int nb = (int) std::min<size_t>(8, jobs.size() - b);
        F->workers->run(nb * 3, [&](int job) {
            const Job& J = jobs[b + (size_t) (job / 3)];
            const int role = job % 3;
            const auto& Ly = F->L[(size_t) J.il];
            const auto& nl = pack_layers_[(size_t) J.il];
            const Shard& sh = pack_shards_[(size_t) (role == 0 ? nl.gate_shard : role == 1 ? nl.up_shard : nl.down_shard)];
            const uint64_t off = role == 0   ? nl.gate_off + (uint64_t) J.e * Ly.gu_bytes
                                 : role == 1 ? nl.up_off + (uint64_t) J.e * Ly.gu_bytes
                                             : nl.down_off + (uint64_t) J.e * Ly.dn_bytes;
            const size_t len = role == 2 ? Ly.dn_bytes : Ly.gu_bytes;
            uint8_t* base = J.vram ? F->stage[(size_t) (job / 3)] : J.dst;
            uint8_t* d = base + (role == 0 ? 0 : role == 1 ? Ly.gu_bytes : 2 * Ly.gu_bytes);
            read_slice(sh, off, len, d);
        });
        for (int k = 0; k < nb; ++k) {
            const Job& J = jobs[b + (size_t) k];
            if (J.vram)
                cudaMemcpyAsync(J.dst, F->stage[(size_t) k], F->L[(size_t) J.il].blob, cudaMemcpyHostToDevice, F->copy);
            done_bytes += F->L[(size_t) J.il].blob;
        }
        cudaStreamSynchronize(F->copy);
        const int pct = (int) (100.0 * (double) done_bytes / (double) std::max<size_t>(1, total_bytes));
        if (pct / 10 != last_pct / 10) {
            last_pct = pct;
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::fprintf(stderr, "glm fast: CUDA%d warming the expert tiers %d%% (%.1f of %.1f GB, %.0f s)\n", dev_, pct,
                         (double) done_bytes / 1e9, (double) total_bytes / 1e9, s);
        }
    }
    if (cudaMemcpy(F->tab, th.data(), th.size() * sizeof(unsigned long long), cudaMemcpyHostToDevice) != cudaSuccess) {
        err = "glm fast: the warmed expert tables did not upload";
        return false;
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "glm fast: CUDA%d tiers warm: %zu experts (%.1f GB) in %.0f s (%.2f GB/s)\n", dev_, jobs.size(),
                 (double) total_bytes / 1e9, s, (double) total_bytes / 1e9 / std::max(1e-9, s));
    return true;
}

void Glm5Model::fast_destroy() {
    FastState* F = fast_;
    if (F == nullptr) return;
    cudaSetDevice(dev_);
    prefill_destroy();
    F->quit.store(true);
    if (F->svc.joinable()) F->svc.join();
    {
        std::lock_guard<std::mutex> lk(F->ah_mu);
        F->ah_quit = true;
    }
    F->ah_cv.notify_all();
    for (auto& t : F->ah_th) t.join();
    F->workers.reset();
    F->cpu_pool.reset();
    if (F->cpu_ans_h) cudaFreeHost(F->cpu_ans_h);
    if (F->cpu_seq_d) cudaFree(F->cpu_seq_d);
    if (F->dcnt_d) cudaFree(F->dcnt_d);
    if (F->timing && F->ah_routes > 0) {
        std::string ov;
        for (int d = 0; d < F->n_ahead; ++d)
            ov += " " + std::to_string(d + 1) + ":" +
                  std::to_string((double) F->ah_ov[d] / (double) std::max<uint64_t>(1, F->ah_npred[d])).substr(0, 4);
        std::string cov;
        for (int d = 0; d < F->n_ahead; ++d)
            cov += " " + std::to_string(d + 1) + ":" +
                   std::to_string(100.0 * (double) F->ah_disk_cov[d] / (double) std::max<uint64_t>(1, F->ah_disk))
                       .substr(0, 4) + "%";
        std::fprintf(stderr,
                     "glm fast ahead CUDA%d: of 8 experts predicted d layers early%s | disk misses %llu, named d early%s, "
                     "by any %.1f%% | reads issued %llu, used %llu (waited %llu, %.1f ms; taken over %llu), queue full %llu, "
                     "no free slot %llu\n",
                     dev_, ov.c_str(), (unsigned long long) F->ah_disk, cov.c_str(),
                     100.0 * (double) F->ah_disk_any / (double) std::max<uint64_t>(1, F->ah_disk),
                     (unsigned long long) F->ah_issued, (unsigned long long) F->ah_used,
                     (unsigned long long) F->ah_waited, (double) F->ah_wait_us.load() / 1000.0,
                     (unsigned long long) F->ah_stolen, (unsigned long long) F->ah_full,
                     (unsigned long long) F->ah_nofree);
    }
    if (F->timing && F->tokens > 0)
        std::fprintf(stderr, "glm fast timing: %llu tokens, %.2f ms/token\n", (unsigned long long) F->tokens,
                     F->ms / (double) F->tokens);
    if (F->timing && F->hits.load() + F->misses.load() > 0)
        std::fprintf(stderr,
                     "glm fast tiers CUDA%d: vram hits %llu, ram fetches %llu, disk reads %llu (%.2f%% vram hit) | "
                     "promotions %llu (scratch %llu) demotions %llu drops %llu prefetches %llu | disk waits %llu, %.1f ms\n",
                     dev_, (unsigned long long) F->hits.load(), (unsigned long long) F->ram_hits.load(),
                     (unsigned long long) F->disk_reads.load(),
                     100.0 * (double) F->hits.load() / (double) std::max<uint64_t>(1, F->hits.load() + F->misses.load()),
                     (unsigned long long) F->promotions.load(), (unsigned long long) F->scratch_uses.load(),
                     (unsigned long long) F->demotions.load(), (unsigned long long) F->drops.load(),
                     (unsigned long long) F->prefetches.load(), (unsigned long long) F->miss_layers.load(),
                     (double) F->miss_us.load() / 1000.0);
    if (F->timing && F->pred_n.load() > 0)
        std::fprintf(stderr, "glm fast predict CUDA%d: %.2f of 8 experts predicted one layer early; %.1f%% of the "
                             "non-resident ones (%llu)\n",
                     dev_, (double) F->pred_overlap.load() / (double) F->pred_n.load(),
                     100.0 * (double) F->pred_miss_hit.load() / (double) std::max<uint64_t>(1, F->pred_miss.load()),
                     (unsigned long long) F->pred_miss.load());
    if (F->prof_on && F->ptokens > 0) {
        std::vector<std::pair<double, std::string>> v;
        double tot = 0;
        for (auto& kv : F->pacc) {
            v.push_back({kv.second, kv.first});
            tot += kv.second;
        }
        std::sort(v.rbegin(), v.rend());
        std::fprintf(stderr, "glm fast prof CUDA%d (%llu tokens, %.2f ms/token of GPU stream time):\n", dev_,
                     (unsigned long long) F->ptokens, tot / (double) F->ptokens);
        for (auto& e : v)
            std::fprintf(stderr, "  %-18s %7.3f ms/token  %5.1f%%\n", e.second.c_str(), e.first / (double) F->ptokens,
                         100.0 * e.first / tot);
    }
    for (auto e : F->pev) cudaEventDestroy(e);
    if (snap_pool_) {
        cudaFree(snap_pool_);
        snap_pool_ = nullptr;
    }
    if (snap_) {
        cudaFree(snap_);
        snap_ = nullptr;
    }
    if (kda_bak_) {
        cudaFree(kda_bak_);
        kda_bak_ = nullptr;
    }
    for (int i = 0; i < 2; ++i) {
        if (spec_hop_h_[i]) cudaFreeHost(spec_hop_h_[i]);
        if (spec_ev_hop_[i]) cudaEventDestroy(spec_ev_hop_[i]);
        spec_hop_h_[i] = nullptr;
        spec_ev_hop_[i] = nullptr;
    }
    if (spec_steps_ > 0)
        std::fprintf(stderr, "glm spec: %llu speculative positions, %.1f%% accepted\n", (unsigned long long) spec_steps_,
                     100.0 * (double) spec_hits_ / (double) spec_steps_);
    if (F->cs) cudaStreamSynchronize(F->cs);
    if (F->copy) cudaStreamSynchronize(F->copy);
    if (F->kda_graphs.captures || F->kda_graphs.replays)
        std::fprintf(stderr, "glm KDA graphs: %llu captures, %llu replays, %llu invalidations\n",
                     (unsigned long long) F->kda_graphs.captures,
                     (unsigned long long) F->kda_graphs.replays,
                     (unsigned long long) F->kda_graphs.invalidations);
    F->kda_graphs.clear();   // before the graph's buffers and streams are freed
    for (auto& R : F->rc)
        if (R.base && R.registered) numa_pinned_free(R.base, (size_t) R.n * R.stride);
        else if (R.base) cudaFreeHost(R.base);
    for (auto& d : F->draining) cudaEventDestroy(d.ev);
    for (auto& m : F->bg) cudaEventDestroy(m.ev);
    for (auto e : F->ev_free) cudaEventDestroy(e);
    for (auto* sb : F->stage)
        if (sb) cudaFreeHost(sb);
    if (F->upd_key_h) cudaFreeHost(F->upd_key_h);
    if (F->upd_val_h) cudaFreeHost(F->upd_val_h);
    if (F->upd_key_d) cudaFree(F->upd_key_d);
    if (F->upd_val_d) cudaFree(F->upd_val_d);
    if (F->ring_h) cudaFreeHost(F->ring_h);
    if (F->resp_h) cudaFreeHost(F->resp_h);
    if (F->emb_h) cudaFreeHost(F->emb_h);
    if (F->hop_h) cudaFreeHost(F->hop_h);
    if (F->tok_h) cudaFreeHost(F->tok_h);
    if (F->scratch) cudaFree(F->scratch);
    if (F->mtp_h) cudaFree(F->mtp_h);
    if (F->gq) cudaFree(F->gq);
    if (F->mtp_catq) cudaFree(F->mtp_catq);
    if (F->mtp_logits) cudaFree(F->mtp_logits);
    if (F->mtp_tok) cudaFree(F->mtp_tok);
    if (F->mtp_tok_h) cudaFreeHost(F->mtp_tok_h);
    if (F->ev_mtp) cudaEventDestroy(F->ev_mtp);
    if (F->pool) cudaFree(F->pool);
    for (uint8_t* p : F->pool_more) cudaFree(p);
    if (F->xpool) cudaFree(F->xpool);
    if (F->arena) cudaFree(F->arena);
    if (F->ps) cudaStreamSynchronize(F->ps);
    if (F->ev_hop) cudaEventDestroy(F->ev_hop);
    if (F->ev_done) cudaEventDestroy(F->ev_done);
    if (F->ev_pred) cudaEventDestroy(F->ev_pred);
    if (F->ev_pf) cudaEventDestroy(F->ev_pf);
    if (F->ev_pf_prev) cudaEventDestroy(F->ev_pf_prev);
    if (F->cs) cudaStreamDestroy(F->cs);
    if (F->copy) cudaStreamDestroy(F->copy);
    if (F->ps) cudaStreamDestroy(F->ps);
    delete F;
    fast_ = nullptr;
}

// the CPU lane's split as a share of a route's RAM-tier experts that go over PCIe (the mean over f = 1..8)
static double plan_share(unsigned long long plan) {
    double s = 0.0;
    for (int f = 1; f <= 8; ++f) s += (double) (f - (int) ((plan >> (4 * f)) & 15ull)) / f;
    return s / 8.0;
}

void Glm5Model::lane_drift_check() {
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        FastState* F = m->fast_;
        if (F == nullptr || F->cpu_pool == nullptr || F->cal_key.empty()) continue;
        const uint64_t e = F->cpu_experts.load(), us = F->cpu_us.load();
        if (e - F->drift_e0 < 2000) continue;
        const double ms = (double) (us - F->drift_us0) / 1000.0 / (double) (e - F->drift_e0);
        F->drift_e0 = e, F->drift_us0 = us;
        if (F->cal_ref <= 0.0) {   // first window after timing: the reference
            F->cal_ref = ms;
            cal_put(F->cal_file, F->cal_key, cal_vals(F->cpu_c_ms, F->cpu_p_ms, F->cpu_ps_ms, ms));
            continue;
        }
        const int dir = ms > 1.15 * F->cal_ref ? 1 : ms < F->cal_ref / 1.15 ? -1 : 0;
        F->drift_n = dir != 0 && dir == F->drift_dir ? F->drift_n + 1 : dir != 0;
        F->drift_dir = dir;
        if (F->drift_n >= 5) {   // 5 requests in a row: the calibration is off, the next start times it
            std::fprintf(stderr, "glm fast: CUDA%d CPU lane: decode %.3f ms an expert vs %.3f at calibration - dropped from "
                                 "%s\n", m->dev_, ms, F->cal_ref, F->cal_file.c_str());
            cal_put(F->cal_file, F->cal_key, "");
            F->cal_key.clear();
        }
    }
}

double Glm5Model::pcie_share() const {
    return fast_ != nullptr && fast_->cpu_pool != nullptr ? plan_share(fast_->cpu_plan) : 1.0;
}

bool Glm5Model::set_pcie_share(double share) {
    bool any = false;
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        FastState* F = m->fast_;
        if (F == nullptr || F->cpu_pool == nullptr) continue;
        unsigned long long plan = F->cpu_plan_start;
        if (share >= 0.0) {
            plan = 0;
            const double s = std::min(1.0, share);
            for (int f = 0; f <= 8; ++f) plan |= (unsigned long long) (f - (int) std::lround(s * f)) << (4 * f);
        }
        F->cpu_plan = plan;   // (the next route reads it; the boundary's background moves follow it too)
        any = true;
    }
    return any;
}

int Glm5Model::cpu_lane_threads() const {
    return fast_ != nullptr && fast_->cpu_pool != nullptr ? fast_->cpu_pool->active() : 0;
}

bool Glm5Model::set_cpu_lane_threads(int n) {
    bool any = false;
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        FastState* F = m->fast_;
        if (F == nullptr || F->cpu_pool == nullptr) continue;
        F->cpu_pool->set_active(n > 0 ? n : F->cpu_pool->size());
        any = true;
    }
    return any;
}

void Glm5Model::set_read_chunks(int n) {
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get())
        if (m->fast_ != nullptr) m->fast_->read_chunks = n > 0 ? std::min(16, n) : 0;
}

Glm5Model::FastStats Glm5Model::fast_stats() const {
    FastStats s;
    for (const Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        const FastState* F = m->fast_;
        if (F == nullptr) continue;
        s.hits += F->hits.load();
        s.misses += F->misses.load();
        s.miss_layers += F->miss_layers.load();
        s.ram_hits += F->ram_hits.load();
        s.disk_reads += F->disk_reads.load();
        s.promotions += F->promotions.load();
        s.cpu_experts += F->cpu_experts.load();
        s.cpu_ms += (double) F->cpu_us.load() / 1000.0;
        s.miss_ms += (double) F->miss_us.load() / 1000.0;
        s.disk_ms += (double) F->disk_us.load() / 1000.0;
        s.pool_slots += F->pool_slots;
        s.pool_gb += (double) F->pool_bytes / 1073741824.0;
        s.ram_gb += (double) F->ram_bytes / 1073741824.0;
        for (const auto& P : F->lp)
            for (char c : P.st) s.pool_used += c == FastState::kResident;
        for (const auto& R : F->rc) {
            s.ram_slots += R.n;
            for (char c : R.st) s.ram_used += c == FastState::kRHold || c == FastState::kRNew;
        }
    }
    if (fast_ != nullptr) {
        s.tokens = fast_->tokens;
        s.ms = fast_->ms;
    }
    return s;
}

// ---------------------------------------------------------------- the service thread
//
// Consumes every route the device publishes, in order: LFU counts, the promotions the device made (an expert
// landed in a spare slot), and - the only case the device waits for - the experts that are on disk only: read
// into a free RAM-tier slot (or pinned staging) and answered with that source for the fetch kernel.
void Glm5Model::fast_service() {
    FastState* F = fast_;
    cudaSetDevice(dev_);
    const Glm5Geometry& g = g_;
    const int K = g.n_exp_used;
    unsigned int next = 1;
    // Idle (through a prompt, between requests) the thread polls the ring with sleeps instead of spinning: it held one
    // core per GPU at 100% while the server waited (2x V100: 200% CPU idle), a core a 6-core PC's prompt lane and the
    // system then lacked, and the server's stall watchdog could never see a stuck engine as idle.  A decode's routes
    // come ~1 ms apart (two GPUs: half a token between a card's routes), so it spins all through one.  After
    // STRATA_GLM_SERVICE_IDLE_MS (200; 0 = spin always) without a route it polls every 0.1 ms (Windows' timer: 1 ms),
    // after 10x that every 2 ms - the first route after a quiet spell is answered up to one sleep later
    static const int kIdleMs = [] {
        const char* v = getenv("STRATA_GLM_SERVICE_IDLE_MS");
        return v ? std::max(0, std::atoi(v)) : 200;
    }();
    auto last_route = std::chrono::steady_clock::now();
    int idle = 0;                        // 0 spinning, 1 short sleeps, 2 long sleeps
    unsigned int polls = 0;
#if defined(_WIN32) && defined(STRATA_USE_HIP)
    auto last_kick = last_route;
    unsigned int kick_polls = 0;
#endif
    while (!F->quit.load(std::memory_order_relaxed)) {
        gf::MoeRequest* rq = F->ring_h + (next % gf::kRingSize);
        const unsigned int sq = rq->seq;
        if (sq != next) {
            // an entry AHEAD of us means the device lapped the ring (only non-waiting routes can be lapped: a
            // disk miss parks the device until it is answered) - resync instead of waiting forever
            if ((int) (sq - next) > 0 && sq % gf::kRingSize == next % gf::kRingSize) {
                F->processed.fetch_add((uint64_t) (sq - next), std::memory_order_release);
                next = sq;
            } else {
#if defined(_WIN32) && defined(STRATA_USE_HIP)
                // Windows' HIP runtime submits queued launches only when the host waits on the GPU (Linux submits
                // each): a route the main thread queued can sit there while this thread waits for it.  After 1 ms
                // without a route, submit what is queued - cudaStreamQuery does without waiting (not while a graph is
                // being captured on the stream).  Submitting every launch instead (GPU_FLUSH_ON_EXECUTION=1) cost
                // ~31 us a launch on a Radeon 8065S: decode 9.8 instead of 15.2 tokens/s
                if ((++kick_polls & 63u) == 0 || idle != 0) {
                    const auto now = std::chrono::steady_clock::now();
                    if (now - last_route > std::chrono::milliseconds(1) && now - last_kick > std::chrono::milliseconds(1)) {
                        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
                        if (cudaStreamIsCapturing(F->cs, &cap) == cudaSuccess && cap == cudaStreamCaptureStatusNone)
                            (void) cudaStreamQuery(F->cs);
                        (void) cudaGetLastError();
                        last_kick = now;
                    }
                }
#endif
                if (idle == 2) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(2));
                } else if (idle == 1) {
#ifdef _WIN32
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
#else
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
#endif
                    if ((++polls & 255u) == 0 &&
                        std::chrono::steady_clock::now() - last_route > std::chrono::milliseconds(10 * kIdleMs))
                        idle = 2;
                } else {
                    cpu_relax();
                    if (kIdleMs > 0 && (++polls & 4095u) == 0 &&
                        std::chrono::steady_clock::now() - last_route > std::chrono::milliseconds(kIdleMs))
                        idle = 1;
                }
                continue;
            }
        }
        if (kIdleMs > 0) {
            idle = 0;
            last_route = std::chrono::steady_clock::now();
        }
        std::atomic_thread_fence(std::memory_order_acquire);
        if (rq->error != 0) {
            F->processed.fetch_add(1, std::memory_order_release);
            ++next;
            continue;
        }
        {
            std::lock_guard<std::mutex> lk(F->mu);
            const int il = rq->layer;
            const unsigned int miss = rq->miss_mask, fetch = rq->fetch_mask, promo = rq->promo_mask;
            const unsigned int cpu = rq->cpu_mask;   // the CPU lane's experts: not in VRAM, computed below
            auto& P = F->lp[(size_t) il];
            auto& R = F->rc[(size_t) F->layer_rc[(size_t) il]];
            ++F->clock;
            int ids[8];
            for (int i = 0; i < K; ++i) ids[i] = rq->ids[i];
            // STRATA_GLM_ROUTE_LOG=<prefix>: "layer e0 .. e7 tier-mask" per route into <prefix>.<device> (cache studies)
            // - the tier masks are fetch, miss and cpu (bits over e0..e7), then " | n0 .. n15": the near misses, the
            //   route's next ranks after the top k, best first
            static const char* rlog = getenv("STRATA_GLM_ROUTE_LOG");
            if (rlog != nullptr) {
                static thread_local FILE* rf = std::fopen((std::string(rlog) + "." + std::to_string(dev_)).c_str(), "w");
                if (rf != nullptr) {
                    std::fprintf(rf, "%d", il);
                    for (int i = 0; i < K; ++i) std::fprintf(rf, " %d", ids[i]);
                    std::fprintf(rf, " %u %u %u |", fetch, miss, cpu);
                    for (int i = 0; i < 16 && rq->near_ids[i] >= 0; ++i) std::fprintf(rf, " %d", (int) rq->near_ids[i]);
                    std::fputc('\n', rf);
                }
            }
            uint64_t nh = 0;
            for (int i = 0; i < K; ++i) {
                const int key = il * g.n_expert + ids[i];
                if (F->cnt[(size_t) key] < (1u << 24)) ++F->cnt[(size_t) key];
                if (++F->usage[(size_t) key] >= (1u << 30))
                    for (auto& u : F->usage) u >>= 1;
                const bool resident = (((miss | fetch | cpu) >> i) & 1u) == 0;
                if (resident) {
                    const int s = F->slot_of[(size_t) key];
                    if (s >= 0) P.tick[(size_t) s] = F->clock;
                    ++nh;
                } else {
                    const int rs0 = F->ram_of[(size_t) key];   // used from the RAM tier: recent there
                    if (rs0 >= 0) R.tick[(size_t) rs0] = F->clock;
                }
                if ((promo >> i) & 1u) {
                    // the device filled one of this layer's spares with it: resident from now on
                    const int s = P.slot_index(rq->promo_ptr[i]);
                    if (s >= 0 && s < P.n) {
                        P.st[(size_t) s] = FastState::kResident;
                        P.key[(size_t) s] = ids[i];
                        P.tick[(size_t) s] = F->clock;
                        F->slot_of[(size_t) key] = s;
                        for (int j = 0; j < gf::kSpares; ++j)
                            if (P.spare[j] == s) P.spare[j] = -1;
                        F->promotions.fetch_add(1, std::memory_order_relaxed);
                    }
                    // exclusive tiers: its RAM copy is released at the next boundary (the fetch reads it now)
                    const int rs = F->ram_of[(size_t) key];
                    if (rs >= 0 && R.st[(size_t) rs] == FastState::kRHold) R.st[(size_t) rs] = FastState::kRRelease;
                } else if (((fetch | miss) >> i) & 1u) {
                    F->scratch_uses.fetch_add(1, std::memory_order_relaxed);
                }
            }
            F->cnt_events += (uint64_t) K;
            if (F->cnt_events >= 32768) {
                F->cnt_events = 0;
                for (auto& c : F->cnt) c >>= 1;
            }
            F->hits.fetch_add(nh, std::memory_order_relaxed);
            {
                // prediction quality: how many of this route's experts (and of its non-resident ones) the previous
                // layer's prediction named
                const auto& pv = F->pred_of[(size_t) il];
                if (pv[0] >= 0) {
                    uint64_t ov = 0, mh = 0;
                    for (int i = 0; i < K; ++i) {
                        bool in = false;
                        for (int q2 = 0; q2 < K; ++q2) in |= pv[(size_t) q2] == ids[i];
                        ov += in;
                        if ((((fetch | miss) >> i) & 1u) && in) ++mh;
                    }
                    F->pred_n.fetch_add(1, std::memory_order_relaxed);
                    F->pred_overlap.fetch_add(ov, std::memory_order_relaxed);
                    F->pred_miss.fetch_add((uint64_t) std::popcount((unsigned) (fetch | miss)), std::memory_order_relaxed);
                    F->pred_miss_hit.fetch_add(mh, std::memory_order_relaxed);
                }
                if (il + 1 < (int) F->pred_of.size())
                    for (int i = 0; i < K; ++i) F->pred_of[(size_t) il + 1][(size_t) i] = rq->pred[i];
            }
            // the next layer's experts this route prefetched into that layer's spares: resident from now on
            const int npf = rq->pf_n;
            if (npf > 0 && il + 1 < (int) F->lp.size()) {
                auto& P1 = F->lp[(size_t) il + 1];
                auto& R1 = F->rc[(size_t) F->layer_rc[(size_t) il + 1]];
                for (int q2 = 0; q2 < npf && q2 < 8; ++q2) {
                    const int e = rq->pf_ids[q2];
                    const int key = (il + 1) * g.n_expert + e;
                    const int s = P1.slot_index(rq->pf_ptr[q2]);
                    if (s < 0 || s >= P1.n) continue;
                    P1.st[(size_t) s] = FastState::kResident;
                    P1.key[(size_t) s] = e;
                    P1.tick[(size_t) s] = F->clock;
                    F->slot_of[(size_t) key] = s;
                    for (int j = 0; j < gf::kSpares; ++j)
                        if (P1.spare[j] == s) P1.spare[j] = -1;
                    const int rs = F->ram_of[(size_t) key];
                    if (rs >= 0 && R1.st[(size_t) rs] == FastState::kRHold) R1.st[(size_t) rs] = FastState::kRRelease;
                    F->prefetches.fetch_add(1, std::memory_order_relaxed);
                }
            }
            F->ram_hits.fetch_add((uint64_t) std::popcount((unsigned) (fetch)), std::memory_order_relaxed);
            F->misses.fetch_add((uint64_t) std::popcount((unsigned) (fetch | miss | cpu)), std::memory_order_relaxed);
            if (F->n_ahead > 0) fast_ahead_route(il, ids, miss, rq->ahead);
            if (miss != 0) {
                // ---- disk only: read each into a free RAM slot (it stays there unless it was promoted) or staging;
                //      one the LOOKAHEAD already read (or is reading) is answered from its RAM slot
                const auto t0 = std::chrono::steady_clock::now();
                const int cls = F->layer_rc[(size_t) il];
                int nm = 0, mi[8], nr = 0, rd[8], nw = 0, wt[8], ns = 0, sl[8];
                uint64_t stv[8];
                uint8_t* dst[8];
                for (int i = 0; i < K; ++i)
                    if ((miss >> i) & 1u) mi[nm++] = i;
                for (int m = 0; m < nm; ++m) {
                    const int key = il * g.n_expert + ids[mi[m]];
                    int rs = F->ram_of[(size_t) key];
                    if (rs >= 0 && (R.st[(size_t) rs] == FastState::kRLoad || R.st[(size_t) rs] == FastState::kRNew)) {
                        dst[m] = R.base + (size_t) rs * R.stride;
                        if (R.st[(size_t) rs] == FastState::kRLoad) {
                            auto& ld = F->rload[(size_t) cls][(size_t) rs];
                            uint64_t v = ld.load(std::memory_order_acquire);
                            if ((v & 3u) == 0u && ld.compare_exchange_strong(v, v | 1u, std::memory_order_acq_rel)) {
                                rd[nr++] = m;   // still queued: read it here (the reader skips it)
                                sl[ns] = rs;
                                stv[ns++] = (v & ~(uint64_t) 3u) | 2u;
                                ++F->ah_stolen;
                            } else if ((v & 3u) != 2u) {
                                wt[nw++] = rs;   // a reader has it in hand
                            }
                            ++F->ah_used;
                        }
                        if ((promo >> mi[m]) & 1u) {   // promoted: a pure staging use, freed at the boundary
                            R.st[(size_t) rs] = FastState::kRRelease;
                            R.key[(size_t) rs] = -1;
                            F->ram_of[(size_t) key] = -1;
                        } else {
                            R.st[(size_t) rs] = FastState::kRNew;
                        }
                        continue;
                    }
                    // Demoted at the last boundary: its RAM copy is still landing (kRDemote until the next boundary makes
                    // it live), so the table has no entry for it - but its bytes are one PCIe copy away, not a disk read
                    // away.  Wait for that drain's event and serve it from the slot.  (The boundary runs only once this
                    // thread has caught up, so F->draining is not being edited here.)  A route that promotes it leaves
                    // a duplicate that the tier clean-up frees.
                    if (rs >= 0 && R.st[(size_t) rs] == FastState::kRDemote && R.key[(size_t) rs] == key) {
                        bool draining = false;
                        for (const auto& d : F->draining)
                            if (d.rclass == cls && d.rslot == rs) {
                                cudaEventSynchronize(d.ev);
                                draining = true;
                                break;
                            }
                        if (draining) {
                            dst[m] = R.base + (size_t) rs * R.stride;
                            continue;
                        }
                    }
                    rd[nr++] = m;
                    rs = -1;
                    for (int s2 = 0; s2 < R.n; ++s2)
                        if (R.st[(size_t) s2] == FastState::kRFree) {
                            rs = s2;
                            break;
                        }
                    if (rs >= 0) {
                        R.key[(size_t) rs] = key;
                        R.tick[(size_t) rs] = F->clock;
                        R.st[(size_t) rs] = ((promo >> mi[m]) & 1u) ? FastState::kRRelease : FastState::kRNew;
                        F->ram_of[(size_t) key] = ((promo >> mi[m]) & 1u) ? -1 : rs;
                        dst[m] = R.base + (size_t) rs * R.stride;
                        if ((promo >> mi[m]) & 1u) R.key[(size_t) rs] = -1;   // a pure staging use: freed at the boundary
                    } else {
                        dst[m] = F->stage[(size_t) m];
                    }
                }
                // each part in chunks across the workers: the O_DIRECT bounce copy of a whole part was ~0.4 ms of the
                // wait (Mercury: 3.6 -> 3.1 ms a disk wait at 8 chunks; STRATA_GLM_READ_CHUNKS overrides - a Windows
                // laptop on an Intel RST RAID read 15% faster with 4, #67 - and setup's calibration tries 4 and 2)
                static const int rch_env = [] {
                    const char* v = getenv("STRATA_GLM_READ_CHUNKS");
                    return std::max(1, std::min(16, v ? std::atoi(v) : 8));
                }();
                const int rch = F->read_chunks > 0 ? F->read_chunks : rch_env;
                F->workers->run(nr * 3 * rch, [&](int job) {
                    const int m = rd[job / (3 * rch)], part = job % (3 * rch);
                    fast_read_part(il, ids[mi[m]], part / rch, dst[m], part % rch, rch);
                });
                for (int i = 0; i < ns; ++i) F->rload[(size_t) cls][(size_t) sl[i]].store(stv[i], std::memory_order_release);
                if (nw > 0) {
                    const auto tw = std::chrono::steady_clock::now();
                    for (int i = 0; i < nw; ++i)
                        while ((F->rload[(size_t) cls][(size_t) wt[i]].load(std::memory_order_acquire) & 3u) != 2u)
                            cpu_relax();
                    F->ah_waited += (uint64_t) nw;
                    F->ah_wait_us.fetch_add((uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
                                                std::chrono::steady_clock::now() - tw).count(),
                                            std::memory_order_relaxed);
                }
                gf::MoeResponse* Rp = F->resp_h;
                Rp->ptr_mask = 0;
                Rp->n_upd = 0;
                Rp->cpu = 0;
                for (int m = 0; m < nm; ++m) {
                    Rp->ptr[mi[m]] = (unsigned long long) dst[m];
                    Rp->ptr_mask |= 1u << mi[m];
                }
                std::atomic_thread_fence(std::memory_order_release);
                Rp->seq = next;
                const uint64_t us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
                                        std::chrono::steady_clock::now() - t0).count();
                F->disk_reads.fetch_add((uint64_t) nr, std::memory_order_relaxed);   // read here (not ahead)
                for (int r2 = 0; r2 < nr; ++r2) ++F->diag_disk[F->left[(size_t) il * g.n_expert + ids[mi[rd[r2]]]] & 3];
                F->disk_us.fetch_add(us, std::memory_order_relaxed);
                F->miss_us.fetch_add(us, std::memory_order_relaxed);
                F->miss_layers.fetch_add(1, std::memory_order_relaxed);
                static const bool trace = getenv("STRATA_GLM_SVC_TRACE") != nullptr;
                if (trace)
                    std::fprintf(stderr, "svc CUDA%d seq %u layer %d disk %d %.2f ms\n", dev_, next, il, nm,
                                 (double) us / 1000.0);
            }
        }
        // ---- the CPU LANE: this route's host experts from their RAM-tier blobs (they stay in RAM until the next
        //      boundary: the device waits for this answer before its down combine, so the token cannot end first)
        if (rq->cpu_mask != 0u && F->cpu_ans_h != nullptr) {
            const auto t0 = std::chrono::steady_clock::now();
            const unsigned int cm = rq->cpu_mask;
            int ne = 0;
            const uint8_t* blob[8];
            float w[8];
            for (int i = 0; i < g.n_exp_used; ++i)
                if ((cm >> i) & 1u) {
                    blob[ne] = (const uint8_t*) rq->cpu_src[i];
                    w[ne++] = rq->w[i];
                }
            fast_cpu_experts(rq->layer, ne, blob, w, rq->x, F->cpu_ans_h->part);
            std::atomic_thread_fence(std::memory_order_release);
            F->cpu_ans_h->seq = next;
            F->cpu_experts.fetch_add((uint64_t) ne, std::memory_order_relaxed);
            F->cpu_routes.fetch_add(1, std::memory_order_relaxed);
            F->cpu_us.fetch_add((uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
                                    std::chrono::steady_clock::now() - t0).count(),
                                std::memory_order_relaxed);
        }
        F->processed.fetch_add(1, std::memory_order_release);
        ++next;
    }
}

// one part (0 gate, 1 up, 2 down) of expert e of layer il - or its chunk-th of n_chunks equal pieces - from the shards
// into its place in the blob at `blob`
void Glm5Model::fast_read_part(int il, int e, int role, uint8_t* blob, int chunk, int n_chunks) {
    const auto& Ly = fast_->L[(size_t) il];
    const auto& nl = pack_layers_[(size_t) il];
    const Shard& sh = pack_shards_[(size_t) (role == 0 ? nl.gate_shard : role == 1 ? nl.up_shard : nl.down_shard)];
    const uint64_t off = role == 0   ? nl.gate_off + (uint64_t) e * Ly.gu_bytes
                         : role == 1 ? nl.up_off + (uint64_t) e * Ly.gu_bytes
                                     : nl.down_off + (uint64_t) e * Ly.dn_bytes;
    const size_t len = role == 2 ? Ly.dn_bytes : Ly.gu_bytes;
    const size_t piece = ((len + (size_t) n_chunks - 1) / (size_t) n_chunks + 4095) & ~(size_t) 4095;
    const size_t c0 = std::min(len, (size_t) chunk * piece), c1 = std::min(len, c0 + piece);
    if (c1 > c0)
        read_slice(sh, off + c0, c1 - c0, blob + (role == 0 ? 0 : role == 1 ? Ly.gu_bytes : 2 * Ly.gu_bytes) + c0);
}

// ---- LOOKAHEAD, for the route of layer il (service thread, F->mu held): score the predictions made for this layer
//      d+1 layers earlier, keep this route's predictions for the layers after it, and queue a read of every
//      predicted expert that is on disk only (VRAM and RAM tiers both lack it) into a free RAM slot.  The slot is
//      kRLoad until it lands; the layer's own route then finds it through ram_of (the device still calls it a disk
//      miss: rtab learns of it at the next boundary).
void Glm5Model::fast_ahead_route(int il, const int* ids, unsigned int miss, const short (*ahead)[8]) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    const int K = g.n_exp_used;
    ++F->ah_routes;
    auto& pr = F->ah_pred[(size_t) il];
    bool cov[8] = {false, false, false, false, false, false, false, false};
    for (int d = 0; d < F->n_ahead; ++d) {
        if (pr[(size_t) d][0] < 0) continue;
        ++F->ah_npred[d];
        for (int i = 0; i < K; ++i) {
            bool in = false;
            for (int q = 0; q < 8; ++q) in |= pr[(size_t) d][(size_t) q] == ids[i];
            F->ah_ov[d] += in;
            if (in && ((miss >> i) & 1u)) {
                ++F->ah_disk_cov[d];
                cov[i] = true;
            }
        }
        pr[(size_t) d].fill(-1);
    }
    for (int i = 0; i < K; ++i)
        if ((miss >> i) & 1u) {
            ++F->ah_disk;
            F->ah_disk_any += cov[i];
        }
    for (int d = 0; d < F->n_ahead; ++d) {
        const int la = il + 1 + d;
        if (ahead[d][0] < 0 || la >= (int) F->ah_pred.size()) continue;
        for (int i = 0; i < 8; ++i) F->ah_pred[(size_t) la][(size_t) d][(size_t) i] = ahead[d][i];
    }
    if (!F->ahead_read) return;
    constexpr int kMaxInflight = 4;
    bool queued = false;
    for (int d = 0; d < F->n_ahead; ++d) {   // the nearest layer first: its reads are the most urgent
        const int la = il + 1 + d;
        if (ahead[d][0] < 0 || la >= (int) F->layer_rc.size() || F->layer_rc[(size_t) la] < 0) continue;
        const int cls = F->layer_rc[(size_t) la];
        auto& R = F->rc[(size_t) cls];
        for (int i = 0; i < K; ++i) {
            const int e = ahead[d][i];
            if (e < 0) continue;
            const int key = la * g.n_expert + e;
            if (F->slot_of[(size_t) key] >= 0 || F->ram_of[(size_t) key] >= 0) continue;
            if (F->ah_inflight.load(std::memory_order_relaxed) >= kMaxInflight) {
                ++F->ah_full;
                continue;
            }
            int rs = -1;
            for (int s = 0; s < R.n; ++s)
                if (R.st[(size_t) s] == FastState::kRFree) {
                    rs = s;
                    break;
                }
            if (rs < 0) {
                ++F->ah_nofree;
                continue;
            }
            R.key[(size_t) rs] = key;
            R.st[(size_t) rs] = FastState::kRLoad;
            F->ram_of[(size_t) key] = rs;
            const uint64_t tag = ++F->ah_tag;
            F->rload[(size_t) cls][(size_t) rs].store(tag << 2, std::memory_order_release);
            F->ah_inflight.fetch_add(1, std::memory_order_relaxed);
            ++F->ah_issued;
            {
                std::lock_guard<std::mutex> lk(F->ah_mu);
                F->ah_q.push_back(FastState::AheadJob{cls, rs, la, e, tag});
            }
            queued = true;
        }
    }
    if (queued) F->ah_cv.notify_all();
}

// the LOOKAHEAD reader threads: one queued expert at a time into the RAM slot the service thread claimed for it - unless
// the route got there first and read it itself (the tag's phase moved on), or the slot was reused under a newer tag
void Glm5Model::fast_ahead_reader() {
    FastState* F = fast_;
    for (;;) {
        FastState::AheadJob j;
        {
            std::unique_lock<std::mutex> lk(F->ah_mu);
            F->ah_cv.wait(lk, [&] { return F->ah_quit || !F->ah_q.empty(); });
            if (F->ah_quit) return;
            j = F->ah_q.front();
            F->ah_q.erase(F->ah_q.begin());
        }
        auto& ld = F->rload[(size_t) j.rclass][(size_t) j.rslot];
        uint64_t want = j.tag << 2;
        if (ld.compare_exchange_strong(want, (j.tag << 2) | 1u, std::memory_order_acq_rel)) {
            uint8_t* blob = F->rc[(size_t) j.rclass].base + (size_t) j.rslot * F->rc[(size_t) j.rclass].stride;
            for (int role = 0; role < 3; ++role) fast_read_part(j.il, j.e, role, blob);
            ld.store((j.tag << 2) | 2u, std::memory_order_release);
        }
        F->ah_inflight.fetch_sub(1, std::memory_order_relaxed);
    }
}

// ---------------------------------------------------------------- between tokens
//
// Every device idle, the service thread caught up: (1) RAM slots of promoted experts free up, disk reads of the
// last token become RAM residents; (2) demotions that landed make their RAM copy live and their VRAM slot free;
// (3) every layer's spares are refilled - from free slots, else by evicting the layer's least-used resident
// (demoted into the RAM tier when it beats the RAM tier's least-used, dropped otherwise); (4) the RAM tier keeps
// a few free slots for disk reads.  The table edits ride one small update kernel at the head of the next token.
bool Glm5Model::fast_boundary(std::string& err) {
    FastState* F = fast_;
    if (F == nullptr || F->lp.empty() || F->rc.empty()) return true;
    // Also catches an asynchronous failure before waiting for routes that may never have been published.
    if (!glmfast::cuda_ok(cudaStreamSynchronize(F->cs), "glm boundary compute", err) || !F->route_ok(err))
        return false;
    const Glm5Geometry& g = g_;
    while (F->processed.load(std::memory_order_acquire) < F->expected) cpu_relax();
    std::lock_guard<std::mutex> lk(F->mu);
    int nu = 0;
    bool ok = true;
    const auto flush = [&] {
        if (nu > 0 && ok) {
            ok = glmfast::cuda_ok(cudaMemcpyAsync(F->upd_key_d, F->upd_key_h, (size_t) nu * sizeof(int),
                                                  cudaMemcpyHostToDevice, F->cs), "glm table keys", err) &&
                 glmfast::cuda_ok(cudaMemcpyAsync(F->upd_val_d, F->upd_val_h,
                                                  (size_t) nu * sizeof(unsigned long long), cudaMemcpyHostToDevice,
                                                  F->cs), "glm table values", err);
            if (ok) {
                gf::tab_update(F->tab, F->upd_key_d, F->upd_val_d, nu, F->cs);
                ok = glmfast::cuda_ok(cudaGetLastError(), "glm table update", err);
            }
            // Pinned sources belong to the DMA until THIS stream completes, including the final batch.
            const cudaError_t e = cudaStreamSynchronize(F->cs);
            if (ok) ok = glmfast::cuda_ok(e, "glm table update sync", err);
        }
        nu = 0;
        ++F->upd_g;
    };
    ++F->upd_g;
    const auto upd = [&](size_t k, unsigned long long v) {
        if (!ok) return;
        if (F->upd_gen[k] == F->upd_g) {   // edited earlier in this batch: the last value is the one that counts
            F->upd_val_h[F->upd_at[k]] = v;
            return;
        }
        if (nu == FastState::kMaxUpd) {   // a full batch goes out first (the host buffers are reused after it ran)
            flush();
            if (!ok) return;
        }
        F->upd_gen[k] = F->upd_g;
        F->upd_at[k] = nu;
        F->upd_key_h[nu] = (int) k;
        F->upd_val_h[nu] = v;
        ++nu;
    };
    // the RAM tier's victim among its held experts: the least-used by the aged counts among those no route used in the
    // last `protect` routes (~64 tokens) - a newly hot expert keeps its place while its count catches up, instead of
    // going back to disk as the lowest count and being read again a moment later (real chats measured 7-11 disk reads
    // a token that way); all of them recent: the least recently used.  STRATA_GLM_RAM_EVICT=lfu: the plain least-used
    // (the old rule), =lru: the least recently used.  -1: none held.
    static const int ram_policy = [] {
        const char* v = getenv("STRATA_GLM_RAM_EVICT");
        return v == nullptr ? 0 : std::strcmp(v, "lfu") == 0 ? 1 : std::strcmp(v, "lru") == 0 ? 2 : 0;
    }();
    static const uint64_t protect = [] {
        const char* v = getenv("STRATA_GLM_RAM_PROTECT");
        return (uint64_t) (v ? std::max(0, std::atoi(v)) : 64) * 42u;
    }();
    const auto ram_victim = [&](const FastState::RamClass& R) -> int {
        int best = -1, oldest = -1;
        uint32_t bc = UINT32_MAX;
        uint64_t bt = UINT64_MAX;
        for (int s = 0; s < R.n; ++s) {
            if (R.st[(size_t) s] != FastState::kRHold) continue;
            const uint64_t t = R.tick[(size_t) s];
            if (t < bt) {
                bt = t;
                oldest = s;
            }
            if (ram_policy == 2) continue;
            if (ram_policy == 0 && t + protect > F->clock) continue;
            const uint32_t c = F->cnt[(size_t) R.key[(size_t) s]];
            if (c < bc || (c == bc && t < R.tick[(size_t) best])) {
                bc = c;
                best = s;
            }
        }
        return best >= 0 ? best : oldest;
    };
    static const bool tier_gc = [] {
        const char* v = getenv("STRATA_GLM_TIER_GC");
        return v == nullptr || std::atoi(v) != 0;
    }();
    // (1)
    for (size_t c = 0; c < F->rc.size(); ++c) {
        auto& R = F->rc[c];
        for (int s = 0; s < R.n; ++s) {
            if (R.st[(size_t) s] == FastState::kRLoad) {
                // a LOOKAHEAD read nobody asked for yet: in the RAM tier once it landed
                if ((F->rload[c][(size_t) s].load(std::memory_order_acquire) & 3u) == 2u) {
                    R.st[(size_t) s] = FastState::kRHold;
                    upd(F->rtab_key(R.key[(size_t) s]), (unsigned long long) (R.base + (size_t) s * R.stride));
                }
            } else if (R.st[(size_t) s] == FastState::kRRelease) {
                const int key = R.key[(size_t) s];
                if (key >= 0) {
                    upd(F->rtab_key(key), 0ull);
                    if (F->ram_of[(size_t) key] == s) F->ram_of[(size_t) key] = -1;
                }
                R.key[(size_t) s] = -1;
                R.st[(size_t) s] = FastState::kRFree;
            } else if (R.st[(size_t) s] == FastState::kRNew) {
                R.st[(size_t) s] = FastState::kRHold;
                upd(F->rtab_key(R.key[(size_t) s]), (unsigned long long) (R.base + (size_t) s * R.stride));
            } else if (R.st[(size_t) s] == FastState::kRHold && tier_gc) {
                // exclusive tiers, kept (STRATA_GLM_TIER_GC=0: not): a held copy the tables lost (a route promoted or re-read the expert while its
                // demotion was still landing, which then held a second copy nothing points at) or of an expert VRAM
                // holds is freed; one that is the expert's only copy is adopted back.  Such copies piled up to ~12%
                // of the experts in no tier at all, the RAM tier full of duplicates (chats: ~5 disk reads a token)
                const int key = R.key[(size_t) s];
                const int ro = key >= 0 ? F->ram_of[(size_t) key] : -1;
                if (key >= 0 && ro == s && F->slot_of[(size_t) key] < 0) continue;   // the normal case
                if (key >= 0 && ro < 0 && F->slot_of[(size_t) key] < 0) {
                    F->ram_of[(size_t) key] = s;
                    upd(F->rtab_key(key), (unsigned long long) (R.base + (size_t) s * R.stride));
                    ++F->diag_adopt;
                    continue;
                }
                if (key >= 0 && (ro == s || ro < 0)) {
                    upd(F->rtab_key(key), 0ull);
                    if (ro == s) F->ram_of[(size_t) key] = -1;
                }
                R.key[(size_t) s] = -1;
                R.st[(size_t) s] = FastState::kRFree;
                ++F->diag_dup;
            }
        }
    }
    // (2)
    std::vector<FastState::Drain> still;
    for (auto& d : F->draining) {
        const cudaError_t de = cudaEventQuery(d.ev);
        if (de != cudaSuccess && de != cudaErrorNotReady)
            return glmfast::cuda_ok(de, "glm demotion event", err);
        if (de == cudaErrorNotReady) {
            still.push_back(d);
            continue;
        }
        auto& R = F->rc[(size_t) d.rclass];
        if (R.st[(size_t) d.rslot] == FastState::kRDemote && R.key[(size_t) d.rslot] == d.key) {
            R.st[(size_t) d.rslot] = FastState::kRHold;
            upd(F->rtab_key(d.key), (unsigned long long) (R.base + (size_t) d.rslot * R.stride));
            if (F->ram_resident) {   // the deferred half of the move: forget the VRAM copy only now
                upd(F->tab_key((size_t) d.key), 0ull);
                F->slot_of[(size_t) d.key] = -1;
            }
        }
        char& vst = F->lp[(size_t) d.il].st[(size_t) d.vslot];
        if (vst == FastState::kDraining) vst = FastState::kFree;   // (a lent slot stays lent)
        F->ev_free.push_back(d.ev);
    }
    F->draining.swap(still);
    // (3)
    for (int il = l0_; il < lt_; ++il) {
        if (F->bg_hold) break;   // lending retires moves without starting spare-refill demotions
        auto& P = F->lp[(size_t) il];
        if (P.n == 0) continue;
        auto& R = F->rc[(size_t) F->layer_rc[(size_t) il]];
        for (int j = 0; j < gf::kSpares; ++j) {
            if (P.spare[j] >= 0) continue;
            int s = -1;
            for (int s2 = 0; s2 < P.n; ++s2)
                if (P.st[(size_t) s2] == FastState::kFree) {
                    s = s2;
                    break;
                }
            if (s >= 0) {
                P.st[(size_t) s] = FastState::kSpare;
                P.key[(size_t) s] = -1;
                P.spare[j] = s;
                upd(F->spare_key(il, j), (unsigned long long) P.slot_ptr(s));
                continue;
            }
            // A full unified pool needs spares only in empty slots (e.g. after
            // prompt lending). Never evict an expert just to reserve a spare.
            if (glmfast::full_unified_pool(F->unified_memory, P.n, g.n_expert)) break;
            // evict the least-used resident of this layer (recency breaks ties).  STRATA_GLM_VRAM_EVICT=lru: the
            // least recently used instead - route traces replayed through these rules (Maya-M, 3090 + 3060 split):
            // the aged counts keep experts that were frequent a while ago, the smaller half's pool most of all (VRAM
            // hits 55.7 -> 65.2 %, off-card experts a token 57.7 -> 45.4 on the 3060; 71.8 -> 64.6 on the 3090)
            static const bool vram_lru = [] {
                const char* ve = getenv("STRATA_GLM_VRAM_EVICT");
                return ve != nullptr && std::strcmp(ve, "lru") == 0;
            }();
            int v = -1;
            uint32_t bc = UINT32_MAX;
            uint64_t bt = UINT64_MAX;
            for (int s2 = 0; s2 < P.n; ++s2) {
                if (P.st[(size_t) s2] != FastState::kResident) continue;
                const uint32_t c = vram_lru ? 0u : F->cnt[(size_t) il * g.n_expert + P.key[(size_t) s2]];
                if (c < bc || (c == bc && P.tick[(size_t) s2] < bt)) {
                    bc = c;
                    bt = P.tick[(size_t) s2];
                    v = s2;
                }
            }
            if (v < 0) break;
            const int vkey = il * g.n_expert + P.key[(size_t) v];
            // demote it into the RAM tier: a free slot, else the RAM tier's victim (if colder than it)
            int rs = -1;
            uint32_t rc_min = UINT32_MAX;
            for (int s2 = 0; s2 < R.n; ++s2)
                if (R.st[(size_t) s2] == FastState::kRFree) {
                    rs = s2;
                    rc_min = 0;
                    break;
                }
            if (rs < 0 && F->ram_resident) {
                // no free RAM slot: keep the VRAM expert (nothing drops to disk); the spare is refilled later
                ++F->diag_resident_skip;
                break;
            }
            // resident mode keeps the VRAM table entry live until the demotion's copy lands (the slot is
            // kDraining, so nothing overwrites it): a route mid-flight still hits VRAM
            if (!F->ram_resident) {
                upd(F->tab_key(vkey), 0ull);
                F->slot_of[(size_t) vkey] = -1;
            }
            ++P.evictions;
            if (rs < 0) {
                rs = ram_victim(R);
                if (rs >= 0) rc_min = F->cnt[(size_t) R.key[(size_t) rs]];
            }
            if (rs >= 0 && (R.st[(size_t) rs] == FastState::kRFree || rc_min < bc)) {
                if (R.st[(size_t) rs] == FastState::kRHold) {
                    const int old = R.key[(size_t) rs];
                    F->left[(size_t) old] = 2;
                    ++F->diag_ram_evict;
                    upd(F->rtab_key(old), 0ull);
                    F->ram_of[(size_t) old] = -1;
                }
                R.key[(size_t) rs] = vkey;
                R.tick[(size_t) rs] = F->clock;
                R.st[(size_t) rs] = FastState::kRDemote;
                F->ram_of[(size_t) vkey] = rs;
                cudaMemcpyAsync(R.base + (size_t) rs * R.stride, P.slot_ptr(v),
                                F->L[(size_t) il].blob, cudaMemcpyDeviceToHost, F->copy);
                cudaEvent_t ev = F->get_event();
                cudaEventRecord(ev, F->copy);
                F->draining.push_back(FastState::Drain{il, v, vkey, F->layer_rc[(size_t) il], rs, ev});
                P.st[(size_t) v] = FastState::kDraining;
                P.key[(size_t) v] = -1;
                F->demotions.fetch_add(1, std::memory_order_relaxed);
            } else {
                // colder than everything in RAM: dropped (disk only); the slot can be a spare right away - the
                // table edit that forgets it runs before any route of the next token
                P.st[(size_t) v] = FastState::kSpare;
                P.key[(size_t) v] = -1;
                P.spare[j] = v;
                upd(F->spare_key(il, j), (unsigned long long) P.slot_ptr(v));
                F->drops.fetch_add(1, std::memory_order_relaxed);
                F->left[(size_t) vkey] = 1;
                ++F->diag_drop;
            }
        }
    }
    // (4) a few free RAM slots per class for the next token's disk reads (and its reads ahead); RAM-resident
    // mode plans no disk reads, so no residents are evicted to keep landing room either
    static const char* kf = getenv("STRATA_GLM_RAM_FREE");
    const int keep_free = F->ram_resident ? 0 : kf ? std::max(1, std::atoi(kf)) : F->ahead_read ? 12 : 4;
    for (auto& R : F->rc) {
        int nfree = 0;
        for (char c : R.st) nfree += c == FastState::kRFree;
        while (nfree < keep_free) {
            const int best = ram_victim(R);
            if (best < 0) break;
            upd(F->rtab_key(R.key[(size_t) best]), 0ull);
            F->ram_of[(size_t) R.key[(size_t) best]] = -1;
            F->left[(size_t) R.key[(size_t) best]] = 2;
            ++F->diag_ram_evict;
            R.key[(size_t) best] = -1;
            R.st[(size_t) best] = FastState::kRFree;
            ++nfree;
        }
    }
    // (5) background moves between the tiers, over the copy stream.  Where the CPU lane takes every RAM-tier expert
    //     (two sockets beside an x8 link: 0.18 ms an expert on the CPU, 2.1 over PCIe) no route fetches one into VRAM
    //     any more, so the pool kept what the warm-up loaded - minus what each prompt's lending moved out (one 3090:
    //     1134 -> 1022 residents after a prompt, 9-15% hits).  (a) Landed moves go live: a promoted expert is
    //     resident and its RAM slot free (exclusive tiers); a demoted one lives in RAM and its VRAM slot is free.
    //     (b) Up to bg_n new moves: a layer's most used RAM-only expert is promoted into a free slot; with none free,
    //     the layer's least used resident is demoted when a RAM-only expert is used bg_ratio times as much (the slot
    //     takes the promotion at a later boundary).  Either way the expert stays live where it was until its copy
    //     landed - no window in which a route would go to disk.  STRATA_GLM_PROMOTE=<moves a boundary>, 0: off.
    {
        const int NE = g.n_expert;
        std::vector<FastState::BgMove> keep;
        for (auto& m : F->bg) {
            const cudaError_t me = cudaEventQuery(m.ev);
            if (me != cudaSuccess && me != cudaErrorNotReady)
                return glmfast::cuda_ok(me, "glm background event", err);
            if (me == cudaErrorNotReady) {
                keep.push_back(m);
                continue;
            }
            auto& P = F->lp[(size_t) m.il];
            auto& R = F->rc[(size_t) m.rclass];
            if (m.up) {
                P.st[(size_t) m.vslot] = FastState::kResident;
                P.key[(size_t) m.vslot] = m.key % NE;
                P.tick[(size_t) m.vslot] = F->clock;
                F->slot_of[(size_t) m.key] = m.vslot;
                upd(F->tab_key((size_t) m.key), (unsigned long long) P.slot_ptr(m.vslot));
                if (R.st[(size_t) m.rslot] == FastState::kRPin) {
                    R.st[(size_t) m.rslot] = FastState::kRFree;
                    R.key[(size_t) m.rslot] = -1;
                }
                if (F->ram_of[(size_t) m.key] == m.rslot) F->ram_of[(size_t) m.key] = -1;
                upd(F->rtab_key((size_t) m.key), 0ull);
                ++F->bg_up;
            } else {
                upd(F->tab_key((size_t) m.key), 0ull);
                if (F->slot_of[(size_t) m.key] == m.vslot) F->slot_of[(size_t) m.key] = -1;
                if (P.st[(size_t) m.vslot] == FastState::kDraining) {
                    P.st[(size_t) m.vslot] = FastState::kFree;
                    P.key[(size_t) m.vslot] = -1;
                }
                R.st[(size_t) m.rslot] = FastState::kRHold;
                R.tick[(size_t) m.rslot] = F->clock;
                F->ram_of[(size_t) m.key] = m.rslot;
                upd(F->rtab_key((size_t) m.key), (unsigned long long) (R.base + (size_t) m.rslot * R.stride));
                ++F->bg_down;
            }
            F->ev_free.push_back(m.ev);
        }
        F->bg.swap(keep);
        static const int bg_n = [] {
            const char* v = getenv("STRATA_GLM_PROMOTE");
            return v != nullptr ? std::max(0, std::atoi(v)) : 8;
        }();
        // (c) the REFILL: while layers have free VRAM slots - a prompt's lending ends with ~15 a layer free on a 24 GB
        //     card - each such layer promotes its hottest RAM-only experts into them, up to STRATA_GLM_PROMOTE_FILL a
        //     boundary in all (24: about a token's worth of the x8 link's copies; the CPU lane loses nothing to the DMA,
        //     a 3090 beside 2x Xeon 6152: 65.5 vs 68.1 GB/s with 6.25 GB/s of H2D beside it) instead of one a layer and
        //     bg_n in all (~80 tokens to refill)
        static const int bg_fill = [] {
            const char* v = getenv("STRATA_GLM_PROMOTE_FILL");
            return v != nullptr ? std::max(0, std::atoi(v)) : 24;
        }();
        const int bg_up_max = std::max(bg_n, bg_fill);
        constexpr uint32_t bg_ratio = 2;
        bool lane_all = F->cpu_plan != 0ull;   // the CPU lane takes every RAM-tier expert of a route
        for (int f = 1; f <= g.n_exp_used && lane_all; ++f) lane_all = (int) ((F->cpu_plan >> (4 * f)) & 15ull) == f;
        if (bg_n > 0 && lane_all && !F->bg_hold && (int) F->bg.size() < 2 * bg_up_max) {
            struct Cand {
                uint32_t hot, cold;
                int il, e, rs, free_slot, cold_slot;
            };
            std::vector<Cand> cands;
            for (int il = l0_; il < lt_; ++il) {
                if (!F->L[(size_t) il].moe || F->layer_rc[(size_t) il] < 0) continue;
                const auto& P = F->lp[(size_t) il];
                const auto& R = F->rc[(size_t) F->layer_rc[(size_t) il]];
                int cs = -1;
                uint32_t cold = UINT32_MAX;
                int frees[8], nfree = 0;   // (refill: up to 8 a layer a boundary)
                for (int s2 = 0; s2 < P.n; ++s2) {
                    if (P.st[(size_t) s2] == FastState::kFree) {
                        if (nfree < (bg_fill > 0 ? 8 : 1)) frees[nfree++] = s2;
                    } else if (P.st[(size_t) s2] == FastState::kResident && P.key[(size_t) s2] >= 0) {
                        const uint32_t c = F->cnt[(size_t) il * NE + P.key[(size_t) s2]];
                        if (c < cold) {
                            cold = c;
                            cs = s2;
                        }
                    }
                }
                // the layer's hottest RAM-only experts: one for each free slot (at least one: the demotion's candidate)
                const int want = std::max(1, nfree);
                std::array<Cand, 8> top{};
                int ntop = 0;
                for (int e = 0; e < NE; ++e) {
                    const size_t key = (size_t) il * NE + e;
                    if (F->slot_of[key] >= 0) continue;
                    const int rs = F->ram_of[key];
                    if (rs < 0 || R.st[(size_t) rs] != FastState::kRHold) continue;
                    const uint32_t h = F->cnt[key];
                    if (ntop == want && h <= top[(size_t) ntop - 1].hot) continue;
                    int at = ntop < want ? ntop++ : ntop - 1;
                    while (at > 0 && top[(size_t) at - 1].hot < h) {
                        top[(size_t) at] = top[(size_t) at - 1];
                        --at;
                    }
                    top[(size_t) at] = Cand{h, cold, il, e, rs, -1, -1};
                }
                for (int j = 0; j < ntop; ++j) {
                    Cand c = top[(size_t) j];
                    c.free_slot = j < nfree ? frees[j] : -1;
                    c.cold_slot = j == 0 ? cs : -1;
                    if (c.free_slot >= 0 || c.cold_slot >= 0) cands.push_back(c);
                }
            }
            std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) { return a.hot > b.hot; });
            int issued = 0, filled = 0;   // demotion moves (at most bg_n), promotions into free slots (bg_up_max)
            for (const auto& c : cands) {
                if (issued >= bg_n && filled >= bg_up_max) break;
                if (c.free_slot >= 0 ? filled >= bg_up_max : issued >= bg_n) continue;
                auto& P = F->lp[(size_t) c.il];
                const int rc = F->layer_rc[(size_t) c.il];
                auto& R = F->rc[(size_t) rc];
                if (c.free_slot >= 0) {
                    P.st[(size_t) c.free_slot] = FastState::kLanding;
                    R.st[(size_t) c.rs] = FastState::kRPin;
                    cudaMemcpyAsync(P.slot_ptr(c.free_slot), R.base + (size_t) c.rs * R.stride, F->L[(size_t) c.il].blob,
                                    cudaMemcpyHostToDevice, F->copy);
                    cudaEvent_t ev = F->get_event();
                    cudaEventRecord(ev, F->copy);
                    F->bg.push_back(FastState::BgMove{c.il, c.free_slot, c.il * NE + c.e, rc, c.rs, true, ev});
                    ++filled;
                } else if (c.cold_slot >= 0 && c.hot >= bg_ratio * std::max<uint32_t>(c.cold, 1u) + 4u) {
                    int rs = -1;
                    for (int s2 = 0; s2 < R.n && rs < 0; ++s2)
                        if (R.st[(size_t) s2] == FastState::kRFree) rs = s2;
                    if (rs < 0) continue;
                    const int vkey = c.il * NE + P.key[(size_t) c.cold_slot];
                    P.st[(size_t) c.cold_slot] = FastState::kDraining;   // (still live in VRAM until the copy landed)
                    R.st[(size_t) rs] = FastState::kRDemote;
                    R.key[(size_t) rs] = vkey;
                    cudaMemcpyAsync(R.base + (size_t) rs * R.stride, P.slot_ptr(c.cold_slot), F->L[(size_t) c.il].blob,
                                    cudaMemcpyDeviceToHost, F->copy);
                    cudaEvent_t ev = F->get_event();
                    cudaEventRecord(ev, F->copy);
                    F->bg.push_back(FastState::BgMove{c.il, c.cold_slot, vkey, rc, rs, false, ev});
                    ++issued;
                }
            }
        }
    }
    flush();
    static const bool diag = getenv("STRATA_GLM_TIER_DIAG") != nullptr;
    if (diag && ++F->diag_b % 64 == 0)
        std::fprintf(stderr, "glm tier diag CUDA%d (%llu boundaries): VRAM drops %llu, RAM evictions %llu, lend drops %llu (to RAM %llu), background promotions %llu / demotions %llu, resident skips %llu | "
                             "disk reads of: never held %llu, dropped from VRAM %llu, evicted from RAM %llu, lend-dropped %llu\n",
                     dev_, (unsigned long long) F->diag_b, (unsigned long long) F->diag_drop,
                     (unsigned long long) F->diag_ram_evict, (unsigned long long) F->diag_lend,
                     (unsigned long long) F->diag_lend_park, (unsigned long long) F->bg_up,
                     (unsigned long long) F->bg_down, (unsigned long long) F->diag_resident_skip,
                     (unsigned long long) F->diag_disk[0], (unsigned long long) F->diag_disk[1],
                     (unsigned long long) F->diag_disk[2], (unsigned long long) F->diag_disk[3]);
    if (diag && F->diag_b % 64 == 0) {
        // per RAM class: slots, held, free, and how many of its layers' experts are in no tier at all
        for (size_t c = 0; c < F->rc.size(); ++c) {
            const auto& R = F->rc[c];
            int held = 0, nfree = 0, out = 0, layers = 0;
            for (char st : R.st) {
                held += st == FastState::kRHold;
                nfree += st == FastState::kRFree;
            }
            for (int il = l0_; il < lt_; ++il) {
                if (!F->L[(size_t) il].moe || F->layer_rc[(size_t) il] != (int) c) continue;
                ++layers;
                for (int e = 0; e < g.n_expert; ++e)
                    out += F->slot_of[(size_t) il * g.n_expert + e] < 0 && F->ram_of[(size_t) il * g.n_expert + e] < 0;
            }
            std::fprintf(stderr, "glm tier diag class %zu (%d layers, %.2f MB): %d slots, %d held, %d free, %d experts in "
                                 "no tier\n", c, layers, (double) R.stride / 1e6, R.n, held, nfree, out);
        }
        std::fprintf(stderr, "glm tier diag clean-up: %llu duplicate RAM copies freed, %llu lost copies adopted\n",
                     (unsigned long long) F->diag_dup, (unsigned long long) F->diag_adopt);
    }
    return ok;
}

// GLM_CB_DIR seam dumps (the reference path's names, so the same diff script bisects both): synchronous,
// debug only - every token overwrites, so a run of n tokens leaves position n-1's seams
static void fast_dump(cudaStream_t s, const std::string& name, const float* dev, int64_t n) {
    static const char* dir = getenv("GLM_CB_DIR");
    if (!dir || !dir[0]) return;
    cudaStreamSynchronize(s);
    std::vector<float> buf((size_t) n);
    cudaMemcpy(buf.data(), dev, (size_t) n * sizeof(float), cudaMemcpyDeviceToHost);
    if (FILE* f = std::fopen((std::string(dir) + "/" + name + ".f32").c_str(), "wb")) {
        std::fwrite(buf.data(), 4, (size_t) n, f);
        std::fclose(f);
    }
}

// ---------------------------------------------------------------- the forward
// ---- one DSA (MLA + indexer) mixer: F->x / F->xq -> F->mixer at position p (the trunk's DSA layers and the NextN block)
bool Glm5Model::fast_dsa(int il, int64_t p, std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    static const bool dumps = getenv("GLM_CB_DIR") != nullptr;
    const auto& Ly = F->L[(size_t) il];
    const float prescale = 1.0f / std::sqrt((float) (g.idx_key * g.idx_heads));
    gf::MvJob j[5];
    j[0] = {Ly.q_a.q, F->xq, F->x, F->qr_raw, nullptr, 1.0f, Ly.q_a.type, g.n_embd, g.q_lora};
    j[1] = {Ly.kv_a.q, F->xq, F->x, F->kv_raw, nullptr, 1.0f, Ly.kv_a.type, g.n_embd, g.kv_lora};
    j[2] = {Ly.idx_k, nullptr, F->x, F->ik_raw, nullptr, 1.0f, gf::kTypeBF16, g.n_embd, g.idx_key};
    j[3] = {Ly.idx_gate, nullptr, F->x, F->ig_raw, nullptr, 1.0f, gf::kTypeBF16, g.n_embd, g.idx_key};
    j[4] = {Ly.idx_proj, nullptr, F->x, F->iw, nullptr, prescale, gf::kTypeBF16, g.n_embd, g.idx_heads};
    if (!gf::mv(j, 5, s)) { err = "glm fast: dsa projections"; return false; }
    if (F->prof_on) F->mark("dsa_proj_mv");
    gf::DsaPrepArgs d;
    d.qr_raw = F->qr_raw;
    d.q_a_norm = Ly.q_a_norm;
    d.qr = F->qr;
    d.qr_q = F->qr_q;
    d.q_lora = g.q_lora;
    d.kv_raw = F->kv_raw;
    d.kv_norm = Ly.kv_a_norm;
    d.lat = (uint16_t*) (state_ + dsa_lat_[(size_t) il]);
    d.kv_lora = g.kv_lora;
    d.lat_q8 = lat_q8_;
    d.ik_raw = F->ik_raw;
    d.k_norm_w = Ly.k_norm_w;
    d.k_norm_b = Ly.k_norm_b;
    d.ik_cache = state_ + dsa_ik_[(size_t) il];
    d.ig_raw = F->ig_raw;
    d.ig_cache = state_ + dsa_ig_[(size_t) il];
    d.ape = Ly.ape;
    d.pooled = state_ + dsa_pool_[(size_t) il];
    d.idx_key = g.idx_key;
    d.kpool = g.idx_kpool;
    d.ring = ik_ring_;
    d.p = (int) p;
    d.eps = g.norm_eps;
    gf::dsa_prep(d, s);
    if (F->prof_on) F->mark("dsa_prep");
    gf::MvJob j2[2];
    j2[0] = {Ly.q_b.q, F->qr_q, nullptr, F->q, nullptr, 1.0f, Ly.q_b.type, g.q_lora, g.n_head * g.qk_nope};
    j2[1] = {Ly.idx_q_b, nullptr, F->qr, F->iq, nullptr, 1.0f, gf::kTypeBF16, g.q_lora, g.idx_heads * g.idx_key};
    if (!gf::mv(j2, 2, s)) { err = "glm fast: dsa q"; return false; }
    if (F->prof_on) F->mark("dsa_q_mv");
    if (dumps) {
        fast_dump(s, "dsa_qr-" + std::to_string(il), F->qr, g.q_lora);
        fast_dump(s, "dsa_q-" + std::to_string(il), F->q, (int64_t) g.n_head * g.qk_nope);
        fast_dump(s, "dsa_iq-" + std::to_string(il), F->iq, (int64_t) g.idx_heads * g.idx_key);
    }
    const int pool_done = (int) ((p + 1) / g.idx_kpool);
    if (pool_done > 0)
        gf::dsa_score(F->iq, state_ + dsa_pool_[(size_t) il], F->iw, g.idx_key, g.idx_heads, pool_done,
                      F->score, s);
    const int top_pools = std::min(g.top_pools_max(), pool_done);
    const int n_sel = g.idx_kpool * top_pools + (g.idx_select_tail ? g.idx_kpool - 1 : 0);
    gf::dsa_select(F->score, pool_done, g.idx_kpool, top_pools, n_sel, (int) p, F->cells, s);
    if (F->prof_on) F->mark("dsa_score_select");
    gf::mla(F->q, Ly.k_b, Ly.v_b, (const uint16_t*) (state_ + dsa_lat_[(size_t) il]), F->cells, n_sel, g.n_head, g.qk_nope,
            g.kv_lora, g.v_head, F->attn_q, s, lat_q8_);
    if (F->prof_on) F->mark("mla");
    gf::MvJob o = {Ly.out.q, F->attn_q, nullptr, F->mixer, nullptr, 1.0f, Ly.out.type,
                   g.n_head * g.v_head, g.n_embd};
    if (!gf::mv(&o, 1, s)) { err = "glm fast: dsa output"; return false; }
    if (F->prof_on) F->mark("dsa_out_mv");
    return true;
}

// ---- one MoE FFN: F->x / F->xq -> F->ffn (routes, tiers, routed + shared experts)
// The NextN draft block leaves its experts that are not in VRAM out (no RAM pull, no disk wait): its draft only has
// to be a good guess - every token is the trunk's own - and its misses sat on the tail's critical path.
// STRATA_GLM_MTP_MISS=1 fetches them like the trunk; STRATA_GLM_MTP_KEEP=<n> fetches the missing ones among the
// route's n best-scored experts only (the route lists them in selection order) and leaves the rest out.  The default
// keeps 2: never slower than skipping all and drafts better (2x V100: 82-83% accepted against 76-81%, decode 28.0-28.2
// against 27.0-28.1 tok/s; RTX 3090 + 3060: 19.5 -> 20.6); KEEP=0 skips every missing one.
static int mtp_skip_from() {
    static const int v = [] {
        if (const char* k = getenv("STRATA_GLM_MTP_KEEP")) return std::max(0, std::min(8, std::atoi(k)));
        return getenv("STRATA_GLM_MTP_MISS") != nullptr ? 8 : 2;
    }();
    return v;
}

// STRATA_GLM_TRUNK_KEEP=<n> (opt-in, changes the answers): the model's own routes leave their experts that are not in
// VRAM out from route rank n on, as the draft block does above - for a RAM-bound split that would rather be faster than
// exact.  The lowest-ranked experts carry the least weight and are the coldest, so they are the ones most often off
// the card.  STRATA_GLM_TRUNK_KEEP_MAP=<digits>: the same per layer, character il for layer il ('1'..'8'; any other
// character, or a layer past the string's end, takes STRATA_GLM_TRUNK_KEEP) - GLM-5.3's early layers notice a left-out
// expert far more than its late ones.  Unset (8): every expert is computed, as before.
static int trunk_skip_from(int il) {
    static const int v = [] {
        const char* k = getenv("STRATA_GLM_TRUNK_KEEP");
        return k != nullptr ? std::max(1, std::min(8, std::atoi(k))) : 8;
    }();
    static const std::string map = [] {
        const char* m = getenv("STRATA_GLM_TRUNK_KEEP_MAP");
        return std::string(m != nullptr ? m : "");
    }();
    if (il >= 0 && il < (int) map.size() && map[(size_t) il] >= '1' && map[(size_t) il] <= '8')
        return map[(size_t) il] - '0';
    return v;
}

bool Glm5Model::fast_moe_lane(int il) const {
    return fast_->cpu_plan != 0ull && fast_->cpu_fmt[(size_t) il].n_ff > 0;
}

// (cpu_wait false: the CPU lane's answer is left for the caller's moe_cpu_wait - the MTP verify runs the other row's
// attention meanwhile)
bool Glm5Model::fast_moe(int il, bool& pf_pending, std::string& err, bool cpu_wait) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    const auto& Ly = F->L[(size_t) il];
    const int FFs = g.n_ff_exp * g.n_shared;
    gf::MvJob j[gf::kMaxMvJobs];
    int nj = 3;
    // a router job: BF16 (GLM's packs) or F32 (hy_v3's, as its GGUF stores it)
    const auto rjob = [&](const FastState::Layer& La, float* y) -> gf::MvJob {
        if (La.router32 != nullptr) return {La.router32, nullptr, F->x, y, nullptr, 1.0f, gf::kTypeF32, g.n_embd, g.n_expert};
        return {La.router, nullptr, F->x, y, nullptr, 1.0f, La.router16 ? gf::kTypeF16 : gf::kTypeBF16, g.n_embd,
                g.n_expert};
    };
    j[0] = rjob(Ly, F->rlog);
    j[1] = {Ly.sh_gate.q, F->xq, nullptr, F->sh_g, nullptr, 1.0f, Ly.sh_gate.type, g.n_embd, FFs};
    j[2] = {Ly.sh_up.q, F->xq, nullptr, F->sh_u, nullptr, 1.0f, Ly.sh_up.type, g.n_embd, FFs};
    // the next layer's router on THIS layer's FFN input: its experts, one layer early
    // (only with prefetch on: the prediction costs a router GEMV and a second top-k per layer)
    const bool pred = F->max_pf > 0 && il + 1 < l1_ && F->L[(size_t) il + 1].moe;
    if (pred)
        j[nj++] = rjob(F->L[(size_t) il + 1], F->plog);
    // LOOKAHEAD: the next layers' routers on this input too (their disk-only experts are read ahead)
    int n_ah = 0;
    const float* ah_bias[gf::kAhead] = {nullptr, nullptr, nullptr, nullptr};
    for (int d = 0; d < F->n_ahead && il + 1 + d < l1_ && F->L[(size_t) il + 1 + d].moe && nj < gf::kMaxMvJobs; ++d) {
        const auto& La = F->L[(size_t) il + 1 + d];
        j[nj++] = rjob(La, F->alog + (size_t) d * g.n_expert);
        ah_bias[d] = La.router_bias;
        ++n_ah;
    }
    if (!gf::mv(j, nj, s)) { err = "glm fast: router / shared expert"; return false; }
    if (F->prof_on) F->mark("router_shexp_mv");
    // the prefetch list of THIS route lives in the buffer of this layer's parity (the side stream may still
    // read the previous layer's list while this route writes)
    gf::MoeDev md = F->md;
    md.pf_src = F->pf_buf[il & 1];
    md.pf_dst = F->pf_buf[il & 1] + 8;
    md.pf_n = F->pf_n_buf[il & 1];
    // the CPU lane splits this layer's RAM-tier experts when the CPU has a dot product for its types
    const bool lane = F->cpu_plan != 0ull && F->cpu_fmt[(size_t) il].n_ff > 0;
    // STRATA_GLM_PROMOTE_MIN=<n>: a fetched expert is kept in VRAM only when its aged route count clears n.
    // Unset or <= 1: the old rule (keep it whenever a spare is free).
    static const int promote_min = [] {
        const char* v = getenv("STRATA_GLM_PROMOTE_MIN");
        return v ? std::max(0, std::atoi(v)) : 0;
    }();
    // STRATA_GLM_PREFETCH_RANK=<n>: prefetch only the prediction's first n ranks (unset: any of its top k)
    static const int pf_rank = [] {
        const char* v = getenv("STRATA_GLM_PREFETCH_RANK");
        return v != nullptr ? std::max(1, std::atoi(v)) : 0;
    }();
    static const int near_n = getenv("STRATA_GLM_ROUTE_LOG") != nullptr ? 16 : 0;   // the route log's near misses
    gf::moe_route(F->rlog, Ly.router_bias, g.n_expert, g.n_exp_used, g.w_scale, g.norm_w != 0, il, F->x,
                  g.n_embd, md, F->sh_g, F->sh_u, g.swiglu_shexp, FFs, F->sh_hq, s,
                  pred ? F->plog : nullptr, pred ? F->L[(size_t) il + 1].router_bias : nullptr,
                  pred ? F->max_pf : 0, n_ah > 0 ? F->alog : nullptr, ah_bias, n_ah,
                  il == mtp_il_ ? mtp_skip_from() : trunk_skip_from(il), lane ? F->cpu_plan : 0ull, promote_min,
                  pf_rank, near_n, g.dsv4() ? 1 : 0,
                  Ly.tid2eid != nullptr && cur_tok_ >= 0 ? Ly.tid2eid + (size_t) cur_tok_ * g.n_exp_used : nullptr);
    if (F->prof_on) F->mark("moe_route");
    ++F->expected;
    // the side stream copies the next layer's predicted experts while this layer computes.  STRATA_GLM_PREFETCH_AT:
    // when that copy starts - "route" (the default: at once, beside this layer's own PCIe fetch and CPU lane),
    // "fetch" (after this layer's own fetch: the link is free) or "cpu" (after this layer's CPU-lane answer: RAM is
    // free too; the copy then runs beside the next layer's attention).  The prefetch lists are double-buffered by
    // layer parity and the next layer waits for this copy before its experts run, so any of the three is safe.
    static const int pf_at = [] {
        const char* v = getenv("STRATA_GLM_PREFETCH_AT");
        return v == nullptr ? 0 : std::strcmp(v, "fetch") == 0 ? 1 : std::strcmp(v, "cpu") == 0 ? 2 : 0;
    }();
    const bool pf = pred && F->max_pf > 0;
    const auto issue_prefetch = [&] {
        cudaEventRecord(F->ev_pred, s);
        cudaStreamWaitEvent(F->ps, F->ev_pred, 0);
        gf::moe_prefetch(md, F->L[(size_t) il + 1].blob, F->ps);
        cudaEventRecord(F->ev_pf, F->ps);
    };
    if (pf && pf_at == 0) issue_prefetch();
    // disk-only experts (rare once warm) park the device until the host has read them; everything not
    // in VRAM is then pulled over PCIe into its slot, and all 8 run from VRAM
    gf::moe_wait(md, g.n_embd, s);
    if (F->prof_on) F->mark("moe_wait");
    gf::moe_fetch(md, g.n_exp_used, Ly.blob, s);
    if (F->prof_on) F->mark("moe_fetch");
    if (pf && pf_at == 1) issue_prefetch();
    // this layer's own prefetched experts (issued one layer ago) must have landed before they are read
    if (pf_pending) cudaStreamWaitEvent(s, F->ev_pf_prev, 0);
    pf_pending = false;
    // (the shared expert's down rides the gate/up launch: sh_out)
    gf::moe_gate_up(Ly.gu_type, md, g.n_exp_used, g.n_embd, g.n_ff_exp, g.swiglu_exp, F->xq, F->hq, Ly.sh_down.q,
                    Ly.sh_down.type, F->sh_hq, FFs, F->sh_out, s);
    if (F->prof_on) F->mark("moe_gate_up");
    gf::moe_down(Ly.d_type, md, g.n_exp_used, g.n_embd, g.n_ff_exp, Ly.down_off, F->hq,
                 Ly.sh_down.q != nullptr ? F->sh_out : nullptr, F->ffn, s);
    if (F->prof_on) F->mark("moe_down");
    // the CPU lane's experts last: the device's own down rows ran while the CPU worked
    if (lane && cpu_wait) {
        gf::moe_cpu_wait(md, g.n_embd, F->ffn, s);
        if (F->prof_on) F->mark("moe_cpu_wait");
    }
    if (pf && pf_at == 2) issue_prefetch();
    if (pf) {
        std::swap(F->ev_pf, F->ev_pf_prev);   // the next layer waits on THIS layer's prefetch
        pf_pending = true;
    }
    return true;
}

// ---- one hy_v3 attention: F->xq (the attn_norm output) -> F->mixer at position p.  q / k / v projections, the
// per-head norms + NeoX rope with the K/V rows into this layer's FP16 caches (gqa_kv_), split-K decode attention over
// [0, p], the output projection
bool Glm5Model::fast_gqa(int il, int64_t p, std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    const auto& Ly = F->L[(size_t) il];
    const int QD = g.n_head * g.head_dim, KD = g.n_kv * g.head_dim;
    gf::MvJob j[3];
    j[0] = {Ly.q.q, F->xq, F->x, F->gq, nullptr, 1.0f, Ly.q.type, g.n_embd, QD};
    j[1] = {Ly.k.q, F->xq, F->x, F->gk, nullptr, 1.0f, Ly.k.type, g.n_embd, KD};
    j[2] = {Ly.v.q, F->xq, F->x, F->gv, nullptr, 1.0f, Ly.v.type, g.n_embd, KD};
    if (!gf::mv(j, 3, s)) { err = "glm fast: gqa projections"; return false; }
    if (F->prof_on) F->mark("gqa_qkv_mv");
    const gf::GqaKv kv = kv_view(il);
    gf::gqa_prep(F->gq, F->gk, F->gv, 1, QD, KD, Ly.q_norm, Ly.k_norm, g.norm_eps, (int) p, g.rope_base, g.n_head, kv, s);
    if (F->prof_on) F->mark("gqa_prep");
    if (kvs_ != nullptr) {   // streamed: the page's summary, the pages to attend to, those cached
        const gf::GqaStream& st = kvs_->lay[(size_t) il];
        gf::gqa_stream_summarize(kv, st, (int) p, (int) p + 1, s);
        gf::gqa_stream_select(F->gq, g.n_head, g.n_kv, st, (int) p + 1, kvs_->budget, kvs_->sink, kvs_->recent,
                              kvs_->sel, s);
        gf::gqa_stream_resolve(st, (int) kvs_->slice, kvs_->sel, s);
        if (F->prof_on) F->mark("gqa_stream");
    }
    if (!gf::gqa_attn(F->gq, kv, (int) p + 1, g.n_head, F->gscr, F->attn_q, s)) {
        err = "glm fast: gqa attention";
        return false;
    }
    if (F->prof_on) F->mark("gqa_attn");
    gf::MvJob o = {Ly.out.q, F->attn_q, nullptr, F->mixer, nullptr, 1.0f, Ly.out.type, QD, g.n_embd};
    if (!gf::mv(&o, 1, s)) { err = "glm fast: gqa output"; return false; }
    if (F->prof_on) F->mark("gqa_out_mv");
    return true;
}

// ---- hy_v3's layers: one residual stream R (= state_), plain pre-norm residuals (rms_q8 adds the block's output
// to R and normalises the next block's input in one launch)
gf::GqaKv Glm5Model::kv_view(int il) const {
    if (kvs_ != nullptr) return kv_host_view(il);
    return gf::gqa_kv_view(state_ + gqa_kv_[(size_t) il], kv_b_, kv_b_ != nullptr ? kv_p0_ : max_ctx_, kv_rec_,
                           kv_loff_[(size_t) il], g_.n_kv, max_ctx_, gqa_kvm_);
}

// ---- KV streaming (glm_kv_stream.hpp).  STRATA_GLM_KV_STREAM=<positions> cached a layer (load_pack: below the
// context); STRATA_GLM_KV_SELECT=<positions> (8192) a decode step attends to by score, STRATA_GLM_KV_RECENT=<positions>
// (1024) the latest ones it always attends to (and the first page).
bool Glm5Model::kv_stream_init(int64_t resident, std::string& err) {
    const Glm5Geometry& g = g_;
    const auto envi = [](const char* k, int64_t d) {
        const char* v = getenv(k);
        return v != nullptr ? std::max<int64_t>(0, std::atoll(v)) : d;
    };
    const int P = gf::kGqaPage;
    auto* K = new KvStream;
    kvs_ = K;
    K->slice = gf::gqa_kv_rec_bytes(g.n_kv, gqa_kvm_);
    K->budget = (int) ((envi("STRATA_GLM_KV_SELECT", 8192) + P - 1) / P);
    K->recent = (int) ((envi("STRATA_GLM_KV_RECENT", 1024) + P - 1) / P) + 1;   // (+ the page being written)
    K->sink = 1;
    const int max_sel = K->budget + K->recent + K->sink + 4;
    const int R = std::max<int>((int) ((resident + P - 1) / P), max_sel + 16);
    const int max_pages = (int) ((max_ctx_ + P - 1) / P);
    if (mtp_il_ >= 0) K->layers.push_back(mtp_il_);
    for (int il = l0_; il < l1_; ++il) K->layers.push_back(il);
    const size_t NL = K->layers.size(), page_b = (size_t) P * K->slice;
    const size_t lay_host = (size_t) max_pages * page_b;
    K->host_bytes = NL * lay_host;
    if (cudaHostAlloc((void**) &K->host, K->host_bytes, cudaHostAllocMapped) != cudaSuccess) {
        cudaGetLastError();
        K->host = nullptr;
        err = "glm kv stream: " + std::to_string(K->host_bytes >> 20) + " MB of pinned host memory did not allocate";
        return false;
    }
    uint8_t* hdev = nullptr;
    if (cudaHostGetDevicePointer((void**) &hdev, K->host, 0) != cudaSuccess) {
        err = "glm kv stream: no device pointer for the host records";
        return false;
    }
    // the device side: per layer the slots, the tables, the summaries; then the selection buffers, the staging copy
    const auto al = [](size_t b) { return (b + 255) / 256 * 256; };
    const size_t summ_b = (size_t) max_pages * g.n_kv * 2 * g.head_dim * 2;
    const size_t lay_dev = al((size_t) R * page_b) + al((size_t) max_pages * 4) + 3 * al((size_t) R * 4) + al(8) + al(summ_b);
    const size_t sel_b = 4 * al((size_t) max_sel * 4) + al(8) + al((size_t) g.n_head * max_pages * 4);
    const size_t stage_b = al(lay_host);
    const size_t dev_b = NL * lay_dev + sel_b + stage_b;
    if (cudaMalloc(&K->dev, dev_b) != cudaSuccess) {
        cudaGetLastError();
        K->dev = nullptr;
        err = "glm kv stream: " + std::to_string(dev_b >> 20) + " MB of VRAM did not allocate";
        return false;
    }
    uint8_t* d = K->dev;
    const auto take = [&](size_t b) { uint8_t* r = d; d += al(b); return r; };
    K->lay.assign((size_t) g.n_layers + 1, gf::GqaStream{});
    for (size_t i = 0; i < NL; ++i) {
        gf::GqaStream& st = K->lay[(size_t) K->layers[i]];
        st.host = hdev + i * lay_host;
        st.slots = take((size_t) R * page_b);
        st.ptab = (int*) take((size_t) max_pages * 4);
        st.spage = (int*) take((size_t) R * 4);
        st.sepoch = (int*) take((size_t) R * 4);
        st.sref = (int*) take((size_t) R * 4);
        st.ctl = (int*) take(8);
        st.summ = take(summ_b);
        st.R = R;
        st.max_pages = max_pages;
        gf::gqa_stream_reset(st, 0);
    }
    K->sel.sel = (int*) take((size_t) max_sel * 4);
    K->sel.miss = (int*) take((size_t) max_sel * 4);
    K->sel.mslot = (int*) take((size_t) max_sel * 4);
    K->sel.nsel = (int*) take(4);
    K->sel.nmiss = (int*) take(4);
    K->sel.ub = (float*) take((size_t) g.n_head * max_pages * 4);
    K->sel.max_sel = max_sel;
    K->stage = take(lay_host);
    cudaDeviceSynchronize();
    std::fprintf(stderr, "glm kv stream: CUDA%d the K/V cache in host memory (%.2f GB pinned, %zu bytes a position and "
                         "layer); in VRAM %.2f GB - %d pages of %d positions a layer cached, the page summaries, a "
                         "staging copy; a decode step attends to %d pages by score + the first + the last %d\n",
                 dev_, (double) K->host_bytes / 1073741824.0, K->slice, (double) dev_b / 1073741824.0, R, P, K->budget,
                 K->recent);
    return true;
}

void Glm5Model::kv_stream_free() {
    if (kvs_ == nullptr) return;
    if (kvs_->dev) cudaFree(kvs_->dev);
    if (kvs_->host) cudaFreeHost(kvs_->host);
    delete kvs_;
    kvs_ = nullptr;
}

gf::GqaKv Glm5Model::kv_host_view(int il) const {
    const gf::GqaStream& st = kvs_->lay[(size_t) il];
    gf::GqaKv kv = gf::gqa_kv_view(st.host, nullptr, max_ctx_, (int64_t) kvs_->slice, 0, g_.n_kv, max_ctx_, gqa_kvm_);
    kv.ptab = st.ptab;
    kv.slots = st.slots;
    kv.sel = kvs_->sel.sel;
    kv.nsel = kvs_->sel.nsel;
    kv.max_sel = kvs_->sel.max_sel;
    return kv;
}

gf::GqaKv Glm5Model::kv_stage_view(int il) const {
    (void) il;
    return gf::gqa_kv_view(kvs_->stage, nullptr, max_ctx_, (int64_t) kvs_->slice, 0, g_.n_kv, max_ctx_, gqa_kvm_);
}

void Glm5Model::kv_stream_reload(int64_t n) {
    if (kvs_ == nullptr) return;
    cudaStream_t s = fast_ != nullptr ? fast_->cs : 0;
    for (int il : kvs_->layers) {
        const gf::GqaStream& st = kvs_->lay[(size_t) il];
        gf::gqa_stream_reset(st, s);
        gf::GqaKv v = kv_host_view(il);
        gf::gqa_stream_summarize(v, st, 0, (int) n, s);
    }
    cudaStreamSynchronize(s);
}

// hy_v3: region b of the K / V records holds positions up to pos_end: the pool's slots under the new part are lent to
// the cache (kKvLent), in steps of 1024 positions (a slot is ~40 positions of every layer) - at most once every ~1000
// tokens of decode
const uint8_t* Glm5Model::kv_lo(int64_t pos_end) const {
    pos_end = std::min(pos_end, max_ctx_);
    return pos_end <= kv_p0_ ? kv_b_ + kv_rec_ : kv_b_ - (pos_end - 1 - kv_p0_) * kv_rec_;
}

size_t Glm5Model::kv_room(int64_t pos_end) const {
    if (fast_ == nullptr) return 0;
    if (kv_b_ == nullptr) return fast_->xpool_bytes;
    const uint8_t* lo = kv_lo(pos_end);
    return lo > fast_->xpool ? (size_t) (lo - fast_->xpool) : 0;
}

bool Glm5Model::kv_ensure(int64_t pos_end, std::string& err) {
    if (kv_b_ == nullptr || pos_end <= kv_lent_) return true;
    const int64_t to = std::min<int64_t>(max_ctx_, (pos_end + 1023) / 1024 * 1024);
    uint64_t moved = 0, dropped = 0;
    if (!lend_range(kv_lo(to), kv_lo(kv_lent_), FastState::kKvLent, moved, dropped, err)) return false;
    kv_lent_ = to;
    return true;
}

// ... and the slots wholly past position keep (rounded up to 1024) go back to the pool, free (the decode's boundaries
// hand them out as spares) - between requests, when the conversation in the model is shorter
void Glm5Model::kv_trim(int64_t keep) {
    FastState* F = fast_;
    if (F == nullptr || kv_b_ == nullptr) return;
    const int64_t to = std::max(kv_p0_, std::min<int64_t>(max_ctx_, (keep + 1023) / 1024 * 1024));
    if (to >= kv_lent_) return;
    const uint8_t* lo = kv_lo(to);   // (region b runs down: the slots wholly below lo go)
    std::lock_guard<std::mutex> lk(F->mu);
    int back = 0;
    for (int il = l0_; il < lt_; ++il) {
        if (!F->L[(size_t) il].moe) continue;
        auto& P = F->lp[(size_t) il];
        for (int s = P.n_main; s < P.n; ++s)
            if (P.st[(size_t) s] == FastState::kKvLent && P.slot_ptr(s) + P.stride <= lo) {
                P.st[(size_t) s] = FastState::kFree;
                ++back;
            }
    }
    // (a slot that straddles `to` stays lent: the lent positions end where its memory ends, at most one slot past to)
    kv_lent_ = to;
    const uint8_t* low = lo;
    for (int il = l0_; il < lt_; ++il) {
        if (!F->L[(size_t) il].moe) continue;
        const auto& P = F->lp[(size_t) il];
        for (int s = P.n_main; s < P.n; ++s)
            if (P.st[(size_t) s] == FastState::kKvLent && P.slot_ptr(s) < low) low = P.slot_ptr(s);
    }
    // the positions whose records lie wholly at or above the lowest lent byte
    if (low <= kv_b_) kv_lent_ = std::max(kv_lent_, std::min<int64_t>(max_ctx_, kv_p0_ + (kv_b_ - low) / kv_rec_ + 1));
    (void) back;
}

gf::Dsv4Layer Glm5Model::d4_layer(int il) const {
    gf::Dsv4Layer L;
    const int r = g_.ratio(il);
    L.ratio = r;
    L.swa = (uint16_t*) (state_ + d4_swa_[(size_t) il]);
    if (r > 0) {
        L.comp = (uint16_t*) (state_ + d4_comp_[(size_t) il]);
        L.st_kv = state_ + d4_st_[(size_t) il];
        L.st_sc = L.st_kv + d4_st_floats(il) / 2;
    }
    if (r == 4) {
        L.icomp = (uint16_t*) (state_ + d4_icomp_[(size_t) il]);
        L.ist_kv = state_ + d4_ist_[(size_t) il];
        L.ist_sc = L.ist_kv + 8 * 2 * g_.idx_key;
    }
    return L;
}

// ---- deepseek4: one attention mixer (glm_dsv4.hpp): the projections of x (f32) / xq (q8_1), the q LoRA's middle,
// the cache update, the indexer's pick (ratio 4, more pooled rows than its top k), the attention, the grouped output
bool Glm5Model::fast_dsv4(int il, int64_t p, std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    const auto& Ly = F->L[(size_t) il];
    const int r = g.ratio(il), HD = g.head_dim, W = r == 4 ? 2 * HD : HD, IW = 2 * g.idx_key;
    gf::MvJob j[gf::kMaxMvJobs];
    int nj = 0;
    j[nj++] = {Ly.q_a.q, F->xq, nullptr, F->d4_qr, nullptr, 1.0f, Ly.q_a.type, g.n_embd, g.q_lora};
    j[nj++] = {Ly.kv_a.q, F->xq, nullptr, F->d4_kv, nullptr, 1.0f, Ly.kv_a.type, g.n_embd, HD};
    if (r > 0) {
        j[nj++] = {Ly.c_kv, nullptr, F->x, F->d4_ckv, nullptr, 1.0f, gf::kTypeF16, g.n_embd, W};
        j[nj++] = {Ly.c_gate, nullptr, F->x, F->d4_csc, nullptr, 1.0f, gf::kTypeF16, g.n_embd, W};
    }
    if (r == 4) {
        j[nj++] = {Ly.ic_kv, nullptr, F->x, F->d4_ickv, nullptr, 1.0f, gf::kTypeF16, g.n_embd, IW};
        j[nj++] = {Ly.ic_gate, nullptr, F->x, F->d4_icsc, nullptr, 1.0f, gf::kTypeF16, g.n_embd, IW};
        j[nj++] = {Ly.idx_proj, nullptr, F->x, F->d4_iw, nullptr, 1.0f, gf::kTypeF16, g.n_embd, g.idx_heads};
    }
    if (!gf::mv(j, nj, s)) { err = "glm fast: deepseek4 projections"; return false; }
    if (F->prof_on) F->mark("d4_proj_mv");
    gf::dsv4_qr_norm(F->d4_qr, Ly.q_a_norm, g.norm_eps, g.q_lora, F->d4_qrn, F->d4_qrn_q8, s);
    nj = 0;
    j[nj++] = {Ly.q_b.q, F->d4_qrn_q8, nullptr, F->d4_q, nullptr, 1.0f, Ly.q_b.type, g.q_lora, g.n_head * HD};
    if (r == 4)
        j[nj++] = {Ly.idx_q_b, nullptr, F->d4_qrn, F->d4_iq, nullptr, 1.0f, gf::kTypeF16, g.q_lora,
                   g.idx_heads * g.idx_key};
    if (!gf::mv(j, nj, s)) { err = "glm fast: deepseek4 q"; return false; }
    if (F->prof_on) F->mark("d4_q_mv");
    const gf::Dsv4Layer L = d4_layer(il);
    const gf::Dsv4Rope& rope = F->d4_rope[(size_t) il];
    const gf::Dsv4KvArgs a{F->d4_kv, Ly.kv_a_norm, F->d4_ckv, F->d4_csc, Ly.c_ape, Ly.c_norm,
                           F->d4_ickv, F->d4_icsc, Ly.ic_ape, Ly.ic_norm, g.norm_eps, g.n_swa};
    gf::dsv4_kv_update(a, L, rope, (int) p, s);
    if (F->prof_on) F->mark("d4_kv");
    const int n_comp = r > 0 ? (int) ((p + 1) / r) : 0;
    const int* sel = nullptr;
    int n_sel = n_comp;
    if (r == 4 && n_comp > g.idx_top_k) {
        gf::dsv4_indexer(F->d4_iq, F->d4_iw, g.idx_heads, L, n_comp, rope, (int) p, g.idx_top_k, F->d4_score, F->d4_sel, s);
        sel = F->d4_sel;
        n_sel = g.idx_top_k;
        if (F->prof_on) F->mark("d4_indexer");
    }
    gf::dsv4_attn(F->d4_q, Ly.sinks, L, g.n_swa, (int) p, sel, n_sel, rope, g.norm_eps, g.n_head, F->d4_heads, s);
    if (F->prof_on) F->mark("d4_attn");
    // the grouped output: group gr's heads (GD values) through wo_a's rows [gr o_lora, (gr + 1) o_lora), then wo_b
    gf::quantize_q8_1(F->d4_heads, F->d4_heads_q8, g.n_head * HD, s);
    const int GD = g.n_head / g.o_groups * HD;
    const size_t wrow = gf::row_bytes(Ly.o_a.type, GD);
    for (int gr = 0; gr < g.o_groups; ++gr)
        j[gr] = {(const uint8_t*) Ly.o_a.q + (size_t) gr * g.o_lora * wrow,
                 (const uint8_t*) F->d4_heads_q8 + (size_t) gr * (GD / 32) * 36, nullptr, F->d4_low + (size_t) gr * g.o_lora,
                 nullptr, 1.0f, Ly.o_a.type, GD, g.o_lora};
    if (!gf::mv(j, g.o_groups, s)) { err = "glm fast: deepseek4 output a"; return false; }
    gf::quantize_q8_1(F->d4_low, F->d4_low_q8, g.o_groups * g.o_lora, s);
    gf::MvJob ob = {Ly.o_b.q, F->d4_low_q8, nullptr, F->mixer, nullptr, 1.0f, Ly.o_b.type, g.o_groups * g.o_lora, g.n_embd};
    if (!gf::mv(&ob, 1, s)) { err = "glm fast: deepseek4 output b"; return false; }
    if (F->prof_on) F->mark("d4_out_mv");
    return true;
}

bool Glm5Model::fast_layers_gqa(int64_t p, std::string& err) {
    FastState* F = fast_;
    if (!kv_ensure(p + 16, err)) return false;   // (+16: the draft block's rows ahead of p)
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    static const bool dumps = getenv("GLM_CB_DIR") != nullptr;
    bool pf_pending = false;
    float* R = state_;
    if (F->prof_on) F->mark("start");
    for (int il = l0_; il < l1_; ++il) {
        const auto& Ly = F->L[(size_t) il];
        gf::rms_q8(R, il == l0_ ? nullptr : F->ffn, Ly.attn_norm, g.norm_eps, g.n_embd, F->x, F->xq, s);
        if (F->prof_on) F->mark("attn_norm");
        if (dumps) {
            if (il != l0_) fast_dump(s, "l_out-" + std::to_string(il - 1), R, g.n_embd);
            fast_dump(s, "attn_norm-" + std::to_string(il), F->x, g.n_embd);
        }
        if (!fast_gqa(il, p, err)) return false;
        gf::rms_q8(R, F->mixer, Ly.ffn_norm, g.norm_eps, g.n_embd, F->x, F->xq, s);
        if (F->prof_on) F->mark("ffn_norm");
        if (dumps) {
            fast_dump(s, "attn_out-" + std::to_string(il), F->mixer, g.n_embd);
            fast_dump(s, "ffn_norm-" + std::to_string(il), F->x, g.n_embd);
        }
        if (!Ly.moe) {
            gf::MvJob j[2];
            j[0] = {Ly.ffn_gate.q, F->xq, nullptr, F->dg, nullptr, 1.0f, Ly.ffn_gate.type, g.n_embd, g.n_ff_dense};
            j[1] = {Ly.ffn_up.q, F->xq, nullptr, F->du, nullptr, 1.0f, Ly.ffn_up.type, g.n_embd, g.n_ff_dense};
            if (!gf::mv(j, 2, s)) { err = "glm fast: dense ffn"; return false; }
            gf::swiglu_q8(F->dg, F->du, g.swiglu_shexp, g.n_ff_dense, F->dhq, s);
            gf::MvJob dn = {Ly.ffn_down.q, F->dhq, nullptr, F->ffn, nullptr, 1.0f, Ly.ffn_down.type, g.n_ff_dense,
                            g.n_embd};
            if (!gf::mv(&dn, 1, s)) { err = "glm fast: dense ffn down"; return false; }
            if (F->prof_on) F->mark("dense_ffn");
        } else if (!fast_moe(il, pf_pending, err)) {
            return false;
        }
        if (dumps) fast_dump(s, "ffn_out-" + std::to_string(il), F->ffn, g.n_embd);
    }
    // the last FFN's residual add (the head reads R); the normalised copy it also writes is not used
    gf::rms_q8(R, F->ffn, F->L[(size_t) l1_ - 1].ffn_norm, g.norm_eps, g.n_embd, F->x, F->xq, s);
    if (dumps) fast_dump(s, "l_out-" + std::to_string(l1_ - 1), R, g.n_embd);
    { cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) { err = std::string("glm fast: ") + cudaGetErrorString(e); return false; } }
    return true;
}

bool Glm5Model::fast_layers(int64_t p, bool hop_in, std::string& err) {
    (void) hop_in;
    if (g_.gqa()) return fast_layers_gqa(p, err);
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    static const bool dumps = getenv("GLM_CB_DIR") != nullptr;
    bool pf_pending = false;
    float* Rc = state_;
    float* Ro = state_ + (int64_t) g.hc * g.n_embd;
    const char* graph_opt = getenv("STRATA_GLM_KDA_GRAPH");
    const bool graphs = graph_opt && std::atoi(graph_opt) != 0 && n_parts_ == 1 && !hop_in &&
                        mtp_il_ < 0 && !F->prof_on && !dumps && F->max_pf == 0 && F->n_ahead == 0 &&
                        getenv("GLM_DUMP_H") == nullptr;
    F->kda_graph_on = graphs;
    // STRATA_GLM_DEFER_CPU=1 (LOSSY, as the GLM tree's 29ca355, after lighttransport/Strata glm53f): a MoE layer's
    // CPU-lane experts are added after the NEXT layer's mixer instead of before its read - the CPU's work overlaps that
    // layer's attention; that attention misses them (the streams get them before its FFN-side read; the last layer waits)
    // (the default on DeepSeek-V4: 2K decode 34.7 / 34.6 -> 28.4 / 26.5 ms a token, 28K 43.0 -> 33.4; KL against the
    // undeferred run 0.009, top-1 94.5 %; needles 3/3; STRATA_GLM_DEFER_CPU=0: off)
    static const bool defer_env = !(getenv("STRATA_GLM_DEFER_CPU") != nullptr && std::atoi(getenv("STRATA_GLM_DEFER_CPU")) == 0);
    const bool defer = defer_env && !graphs && !hop_in && F->cpu_plan != 0ull;
    if (defer && F->defer_post == nullptr && cudaMalloc(&F->defer_post, 8 * sizeof(float)) != cudaSuccess) {
        err = "glm fast: the deferred post weights did not allocate";
        return false;
    }
    bool deferred = false;   // the previous layer's CPU answer is still out
    if (F->prof_on) F->mark("start");
    for (int il = l0_; il < l1_; ++il) {
        const auto& Ly = F->L[(size_t) il];
        const auto body = [&]() -> bool {
        // ---- the attention-side read (fused with the previous FFN's write)
        gf::HcArgs h;
        h.block_out = il == l0_ ? nullptr : F->ffn;
        h.R_old = Rc;
        h.R_new = Ro;
        h.post_in = F->post;
        h.comb_in = F->comb;
        h.pre = F->pre;
        h.post = F->post;
        h.comb = F->comb;
        h.w_fn = Ly.hc_attn_fn;
        h.w_f16 = g.dsv4();
        h.w_scale = Ly.hc_attn_scale;
        h.w_base = Ly.hc_attn_base;
        h.norm_w = Ly.attn_norm;
        h.norm_eps = g.norm_eps;
        h.hc_eps = g.hc_eps;
        h.iters = g.sinkhorn_iters;
        h.n_embd = g.n_embd;
        h.x = F->x;
        h.xq = F->xq;
        h.part = F->part;
        h.counter = F->counter;
        gf::hc(h, s);
        if (F->prof_on) F->mark("hc_attn");
        if (il != l0_) std::swap(Rc, Ro);
        if (dumps) {
            const std::string Ls = std::to_string(il);
            if (il != l0_) fast_dump(s, "l_out-" + std::to_string(il - 1), Rc, (int64_t) g.hc * g.n_embd);
            fast_dump(s, "attn_norm-" + Ls, F->x, g.n_embd);
        }

        if (Ly.recr) {
            const int DI = g.d_inner();
            gf::MvJob j[6];
            j[0] = {Ly.q.q, F->xq, nullptr, F->proj[0], nullptr, 1.0f, Ly.q.type, g.n_embd, DI};
            j[1] = {Ly.k.q, F->xq, nullptr, F->proj[1], nullptr, 1.0f, Ly.k.type, g.n_embd, DI};
            j[2] = {Ly.v.q, F->xq, nullptr, F->proj[2], nullptr, 1.0f, Ly.v.type, g.n_embd, DI};
            j[3] = {Ly.f_a, nullptr, F->x, F->fa, nullptr, 1.0f, gf::kTypeBF16, g.n_embd, g.kda_head_dim};
            j[4] = {Ly.g_a, nullptr, F->x, F->ga, nullptr, 1.0f, gf::kTypeBF16, g.n_embd, g.kda_head_dim};
            j[5] = {Ly.beta, nullptr, F->x, F->beta, nullptr, 1.0f, gf::kTypeBF16, g.n_embd, g.n_head};
            if (!gf::mv(j, 6, s)) { err = "glm fast: kda projections"; return false; }
            if (F->prof_on) F->mark("kda_proj_mv");
            gf::KdaPrepArgs k;
            for (int i = 0; i < 3; ++i) {
                k.proj[i] = F->proj[i];
                k.conv_w[i] = Ly.conv[i];
                k.out[i] = F->conv[i];
            }
            k.conv_state = state_ + kda_conv_[(size_t) il];
            k.fa = F->fa;
            k.ga = F->ga;
            k.f_b = Ly.f_b;
            k.g_b = Ly.g_b;
            k.dt_bias = Ly.dt_bias;
            k.ssm_a = Ly.ssm_a;
            k.lower_bound = g.kda_lb;
            k.g1 = F->g1;
            k.g2 = F->g2;
            k.n_head = g.n_head;
            k.head_dim = g.kda_head_dim;
            k.d_conv = g.d_conv;
            gf::kda_prep(k, s);
            if (F->prof_on) F->mark("kda_prep");
            gf::kda_rec(F->conv[0], F->conv[1], F->conv[2], F->g1, F->beta, state_ + kda_S_[(size_t) il], F->g2,
                        Ly.ssm_norm, g.norm_eps, g.n_head, g.kda_head_dim, F->gated_q, s);
            if (F->prof_on) F->mark("kda_rec");
            gf::MvJob o = {Ly.out.q, F->gated_q, nullptr, F->mixer, nullptr, 1.0f, Ly.out.type, DI, g.n_embd};
            if (!gf::mv(&o, 1, s)) { err = "glm fast: kda output"; return false; }
            if (F->prof_on) F->mark("kda_out_mv");
        } else if (Ly.dsv4) {
            if (!fast_dsv4(il, p, err)) return false;
        } else {
            if (!fast_dsa(il, p, err)) return false;
        }

        if (deferred) {   // the previous layer's CPU experts, into the streams before they are read again
            gf::moe_cpu_wait_streams(F->md, g.n_embd, F->defer_post, Rc, g.hc, s);
            deferred = false;
            if (F->prof_on) F->mark("moe_cpu_wait_late");
        }
        // ---- the FFN-side read (fused with the mixer's write)
        h.block_out = F->mixer;
        h.R_old = Rc;
        h.R_new = Ro;
        h.w_fn = Ly.hc_ffn_fn;
        h.w_scale = Ly.hc_ffn_scale;
        h.w_base = Ly.hc_ffn_base;
        h.norm_w = Ly.ffn_norm;
        gf::hc(h, s);
        if (F->prof_on) F->mark("hc_ffn");
        std::swap(Rc, Ro);
        if (dumps) {
            const std::string Ls = std::to_string(il);
            fast_dump(s, "hc_attn_post-" + Ls, Rc, (int64_t) g.hc * g.n_embd);
            fast_dump(s, "ffn_norm-" + Ls, F->x, g.n_embd);
            fast_dump(s, "mixer-" + Ls, F->mixer, g.n_embd);
        }

        if (!Ly.moe) {
            gf::MvJob j[2];
            j[0] = {Ly.ffn_gate.q, F->xq, nullptr, F->dg, nullptr, 1.0f, Ly.ffn_gate.type, g.n_embd, g.n_ff_dense};
            j[1] = {Ly.ffn_up.q, F->xq, nullptr, F->du, nullptr, 1.0f, Ly.ffn_up.type, g.n_embd, g.n_ff_dense};
            if (!gf::mv(j, 2, s)) { err = "glm fast: dense ffn"; return false; }
            if (F->prof_on) F->mark("dense_gu_mv");
            gf::swiglu_q8(F->dg, F->du, g.swiglu_shexp, g.n_ff_dense, F->dhq, s);
            if (F->prof_on) F->mark("dense_swiglu");
            gf::MvJob dn = {Ly.ffn_down.q, F->dhq, nullptr, F->ffn, nullptr, 1.0f, Ly.ffn_down.type, g.n_ff_dense,
                            g.n_embd};
            if (!gf::mv(&dn, 1, s)) { err = "glm fast: dense ffn down"; return false; }
            if (F->prof_on) F->mark("dense_down_mv");
        } else {
            const bool def = defer && il + 1 < l1_ && F->cpu_fmt[(size_t) il].n_ff > 0;
            if (!fast_moe(il, pf_pending, err, !def)) return false;
            if (def) {   // this layer's FFN-side post weights (the next read overwrites them)
                cudaMemcpyAsync(F->defer_post, F->post, 8 * sizeof(float), cudaMemcpyDeviceToDevice, s);
                deferred = true;
            }
        }
        if (dumps) fast_dump(s, "ffn_out-" + std::to_string(il), F->ffn, g.n_embd);
        return true;
        };
        if (graphs && Ly.recr) {
            bool replayed = false;
            const glmfast::LayerGraphs::Key key{F->cpu_plan, Rc, Ro};
            if (!F->kda_graphs.enqueue(il, key, s, F->expected, Ly.moe ? 1 : 0, body, replayed, err))
                return false;
            // Capture executes the host pointer swaps; replay only executes device
            // work. Maintain the identical residual convention in both cases.
            if (replayed) {
                if (il != l0_) std::swap(Rc, Ro);
                std::swap(Rc, Ro);
            }
        } else if (!body()) {
            return false;
        }
    }
    // the last layer's write half: R (in the OTHER buffer) = post x ffn + comb . R
    gf::hc_post(F->ffn, Rc, F->post, F->comb, g.n_embd, Ro, s);
    if (F->prof_on) F->mark("hc_post_end");
    // keep the convention "the residual lives at state_" for the hop and the head: copy back
    if (Ro != state_)
        cudaMemcpyAsync(state_, Ro, (size_t) g.hc * g.n_embd * sizeof(float), cudaMemcpyDeviceToDevice, s);
    { cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) { err = std::string("glm fast: ") + cudaGetErrorString(e); return false; } }
    return true;
}

bool Glm5Model::fast_token(int32_t token, std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    const int64_t p = pos_++;
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) m->cur_tok_ = token;
    // between tokens (every device idle): finished promotions go live in the device tables
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        cudaSetDevice(m->dev_);
        if (!m->fast_boundary(err)) return false;
    }
    cudaSetDevice(dev_);
    // ---- the embedding row (host-dequantized from the shard mapping) -> the 4 streams
    const ggml_type_traits* tt = ggml_get_type_traits((ggml_type) pack_emb_type_);
    if (tt == nullptr || tt->to_float == nullptr || pack_emb_src_ == nullptr) {
        err = "glm fast: no embedding dequantizer";
        return false;
    }
    tt->to_float(pack_emb_src_ + (size_t) token * ggml_row_size((ggml_type) pack_emb_type_, g.n_embd), F->emb_h,
                 g.n_embd);
    if (const float* img = image_row(p)) std::memcpy(F->emb_h, img, (size_t) g.n_embd * sizeof(float));   // an image
    cudaMemcpyAsync(F->emb, F->emb_h, (size_t) g.n_embd * sizeof(float), cudaMemcpyHostToDevice, F->cs);
    if (g.gqa())   // one residual stream
        cudaMemcpyAsync(state_, F->emb, (size_t) g.n_embd * sizeof(float), cudaMemcpyDeviceToDevice, F->cs);
    else
        gf::embed_streams(F->emb, state_, g.n_embd, F->cs);
    if (!fast_layers(p, false, err)) return false;
    // each later part of a split: the residual crosses over through the previous part's pinned hop buffer, the
    // copy ordered on the devices by an event (the host never waits between the parts)
    Glm5Model* tail = this;
    for (Glm5Model* B = split_next_.get(); B != nullptr; B = B->split_next_.get()) {
        FastState* FA = tail->fast_;
        FastState* FB = B->fast_;
        const size_t hop = (size_t) g.hc * g.n_embd * sizeof(float);
        cudaSetDevice(tail->dev_);
        cudaMemcpyAsync(FA->hop_h, tail->state_, hop, cudaMemcpyDeviceToHost, FA->cs);
        cudaEventRecord(FA->ev_hop, FA->cs);
        cudaSetDevice(B->dev_);
        B->pos_ = pos_;
        cudaStreamWaitEvent(FB->cs, FA->ev_hop, 0);
        cudaMemcpyAsync(B->state_, FA->hop_h, hop, cudaMemcpyHostToDevice, FB->cs);
        if (!B->fast_layers(p, true, err)) return false;
        tail = B;
    }
    FastState* FT = tail->fast_;
    cudaSetDevice(tail->dev_);
    if (g.gqa())
        gf::rms_q8(tail->state_, nullptr, tail->w_.at("output_norm.weight"), g.norm_eps, g.n_embd, FT->head_x,
                   FT->head_xq, FT->cs);
    else if (g.dsv4())
        gf::dsv4_hc_head(tail->state_, tail->w16_.at("output_hc_fn.weight"), tail->w_.at("output_hc_scale.weight"),
                         tail->w_.at("output_hc_base.weight"), g.hc_eps, tail->w_.at("output_norm.weight"), g.norm_eps,
                         g.n_embd, FT->head_x, FT->head_xq, FT->cs);
    else
        gf::head_prep(tail->state_, tail->w_.at("output_norm.weight"), g.norm_eps, g.n_embd, FT->head_x, FT->head_xq,
                      FT->cs);
    // GLM_DUMP_H=<file> (debug): append every position's final hidden state (output_norm of the stream mean - what the
    // NextN/MTP block reads) as n_embd floats
    static const char* dh = getenv("GLM_DUMP_H");
    if (dh != nullptr && dh[0]) {
        cudaStreamSynchronize(FT->cs);
        std::vector<float> hb((size_t) g.n_embd);
        cudaMemcpy(hb.data(), FT->head_x, hb.size() * sizeof(float), cudaMemcpyDeviceToHost);
        if (FILE* f = std::fopen(dh, "ab")) {
            std::fwrite(hb.data(), sizeof(float), hb.size(), f);
            std::fclose(f);
        }
    }
    const WSlot& ow = tail->ws_map_.at("output.weight");
    gf::MvJob o = {ow.q, FT->head_xq, FT->head_x, tail->sc_ + tail->sc_logits, nullptr, 1.0f, ow.type, g.n_embd,
                   g.n_vocab};
    if (ow.type == 0) o.w = ow.f32;
    if (!gf::mv(&o, 1, FT->cs)) {
        err = "glm fast: output head (type " + std::to_string(ow.type) + ")";
        return false;
    }
    gf::argmax(tail->sc_ + tail->sc_logits, g.n_vocab, FT->tok, FT->cs);
    cudaMemcpyAsync(FT->tok_h, FT->tok, sizeof(int), cudaMemcpyDeviceToHost, FT->cs);
    cudaEventRecord(FT->ev_done, FT->cs);
    // wait for the token (the service threads answer misses meanwhile)
    const auto tw = std::chrono::steady_clock::now();
    bool warned = false;
    cudaError_t done;
    while ((done = cudaEventQuery(FT->ev_done)) == cudaErrorNotReady) {
        std::this_thread::yield();
        if (!warned && std::chrono::steady_clock::now() - tw > std::chrono::seconds(10)) {
            warned = true;
            for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
                const FastState* Fm = m->fast_;
                unsigned int last = 0;
                for (int i = 0; i < gf::kRingSize; ++i) last = std::max(last, (unsigned int) Fm->ring_h[i].seq);
                std::fprintf(stderr, "glm fast: token %lld waiting > 10 s - CUDA%d last route seq %u, last answer %u\n",
                             (long long) p, m->dev_, last, (unsigned int) Fm->resp_h->seq);
            }
        }
    }
    if (!glmfast::cuda_ok(done, "glm token completion", err)) return false;
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get())
        if (!m->fast_->route_ok(err)) return false;
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        err = std::string("glm fast: ") + cudaGetErrorString(e);
        return false;
    }
    if (gf::launch_errors() > 0) {
        err = "glm fast: " + std::to_string(gf::launch_errors()) + " kernel launch(es) failed (see stderr)";
        return false;
    }
    last_tok_ = FT->tok_h[0];
    if (d4mtp_ != nullptr && split_next_ == nullptr && d4spec_ == nullptr) {
        // the MTP block's shadow draft: did the last one call this token, then the next one
        if (mtp_pending_ >= 0 && mtp_pending_pos_ == p + 1) {
            ++mtp_tries_;
            mtp_hits_ += mtp_pending_ == last_tok_;
        }
        int d = -1;
        if (!d4_mtp_draft(last_tok_, p + 1, d, err)) return false;
        mtp_pending_ = d;
        mtp_pending_pos_ = p + 2;
        if (mtp_tries_ > 0 && mtp_tries_ % 64 == 0 && mtp_pending_ >= 0)
            std::fprintf(stderr, "dsv4 mtp: %llu / %llu drafts accepted (%.1f%%), %.2f ms a draft\n",
                         (unsigned long long) mtp_hits_, (unsigned long long) mtp_tries_,
                         100.0 * mtp_hits_ / mtp_tries_, mtp_ms_ / (mtp_tries_ + 1));
    }
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get())
        if (m->fast_ && m->fast_->prof_on) {
            cudaSetDevice(m->dev_);
            m->fast_->collect();
        }
    cudaSetDevice(dev_);
    return true;
}

// ---------------------------------------------------------------- the NextN draft block
// At position p (whose final hidden state the head left in head_x): eh_proj([enorm(emb(next)), hnorm(h_p)]) -> a DSA
// mixer and a MoE FFN with plain pre-norm residuals (the trunk's own kernels: fast_dsa / fast_moe on the block's layer
// index; its caches at p) -> shared_head_norm -> the output head -> argmax into mtp_tok (mtp_tok_h after ev_mtp).
bool Glm5Model::fast_mtp(int64_t p, int32_t next_tok, std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    const int il = mtp_il_;
    if (F == nullptr || il < 0) {
        err = "glm mtp: no draft block on this half";
        return false;
    }
    const auto& Ly = F->L[(size_t) il];
    const int E = g.n_embd;
    const ggml_type_traits* tt = ggml_get_type_traits((ggml_type) pack_emb_type_);
    if (tt == nullptr || tt->to_float == nullptr || pack_emb_src_ == nullptr) {
        err = "glm mtp: no embedding dequantizer";
        return false;
    }
    if (!glmfast::cuda_ok(cudaStreamSynchronize(s), "glm mtp embedding sync", err) || !F->route_ok(err))
        return false;   // emb_h may still feed an earlier copy
    tt->to_float(pack_emb_src_ + (size_t) next_tok * ggml_row_size((ggml_type) pack_emb_type_, E), F->emb_h, E);
    cudaMemcpyAsync(F->emb, F->emb_h, (size_t) E * sizeof(float), cudaMemcpyHostToDevice, s);
    gf::mtp_in(F->emb, F->head_x, Ly.enorm, Ly.hnorm, g.norm_eps, E, F->mtp_catq, s);
    gf::MvJob eh = {Ly.eh.q, F->mtp_catq, nullptr, F->mtp_h, nullptr, 1.0f, Ly.eh.type, 2 * E, E};
    if (!gf::mv(&eh, 1, s)) {
        err = "glm mtp: eh_proj";
        return false;
    }
    gf::rms_q8(F->mtp_h, nullptr, Ly.attn_norm, g.norm_eps, E, F->x, F->xq, s);
    if (!(Ly.gqa ? fast_gqa(il, p, err) : fast_dsa(il, p, err))) return false;
    gf::rms_q8(F->mtp_h, F->mixer, Ly.ffn_norm, g.norm_eps, E, F->x, F->xq, s);
    bool pf_pending = false;
    if (!fast_moe(il, pf_pending, err)) return false;
    gf::rms_q8(F->mtp_h, F->ffn, Ly.shnorm, g.norm_eps, E, F->head_x, F->head_xq, s);
    const WSlot& ow = ws_map_.at("output.weight");
    gf::MvJob o = {ow.q, F->head_xq, F->head_x, F->mtp_logits, nullptr, 1.0f, ow.type, E, g.n_vocab};
    if (ow.type == 0) o.w = ow.f32;
    if (!gf::mv(&o, 1, s)) {
        err = "glm mtp: the head";
        return false;
    }
    gf::argmax(F->mtp_logits, g.n_vocab, F->mtp_tok, s);
    cudaMemcpyAsync(F->mtp_tok_h, F->mtp_tok, sizeof(int), cudaMemcpyDeviceToHost, s);
    cudaEventRecord(F->ev_mtp, s);
    return true;
}

// A sampled (not greedy) token from this half's logits, drawn on its own stream and read back through the pinned
// token word.  On the default stream the draw ended in a device-wide sync, which also waited for the tier copies in
// flight on the copy stream every sampled token - and on Windows / HIP lost the prompt after the first token (#6,
// found by jerem91150).  The stream holds this position's forward and nothing after it (the split's tail drains
// before it takes its next position), so the logits read are the ones just computed.
int Glm5Model::fast_sample(strata::kernels::SamplerParams& sp, std::string& err) {
    FastState* F = fast_;
    cudaSetDevice(dev_);
    strata::kernels::sample_tokens(sc_ + sc_logits, 1, (int) g_.n_vocab, nullptr, 0, sp, d_tok_, F->cs);
    cudaMemcpyAsync(F->tok_h, d_tok_, sizeof(int), cudaMemcpyDeviceToHost, F->cs);
    if (!glmfast::cuda_ok(cudaStreamSynchronize(F->cs), "glm sampler", err) || !F->route_ok(err)) return -1;
    return F->tok_h[0];
}

// the bytes the expert pool may take from what is free: headroom kept (cuBLAS-free path; the context's own growth is
// already allocated in the state arena) - ~700 MB for the driver, the sampler and kernel launches' local memory,
// STRATA_GLM_RESERVE_MB; STRATA_GLM_POOL_GB caps it; STRATA_GLM_VRAM_GB=<n> behaves like a card with n GB (the cap
// counts everything this process already holds on the device, so the pool gets what such a card would have).  The
// prompt path and the layer split search size from the same number, before the pool is carved (a discrete card's:
// fast_setup prices a unified-memory APU's pool itself).
size_t Glm5Model::pool_avail(size_t free_b, size_t total_b) {
    size_t reserve = (size_t) 700 << 20;
    if (const char* r = getenv("STRATA_GLM_RESERVE_MB")) reserve = (size_t) std::atoll(r) << 20;
    size_t avail = glmfast::expert_pool_budget(false, free_b, 0, reserve, 0, 0);
    if (const char* cap = getenv("STRATA_GLM_POOL_GB"))
        avail = std::min(avail, (size_t) (std::atof(cap) * 1073741824.0));
    if (const char* vc = getenv("STRATA_GLM_VRAM_GB")) {
        const size_t cap = (size_t) (std::atof(vc) * 1073741824.0);
        const size_t used = total_b - free_b;
        avail = std::min(avail, cap > used + reserve ? cap - used - reserve : (size_t) 0);
    }
    return avail;
}

bool Glm5Model::has_mtp() const {
    for (const Glm5Model* m = this; m != nullptr; m = m->split_next_.get())
        if (m->fast_ != nullptr && m->mtp_il_ >= 0) return true;
    return false;
}

int Glm5Model::mtp_draft(int32_t next_tok, std::string& err) {
    Glm5Model* t = nullptr;
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get())
        if (m->fast_ != nullptr && m->mtp_il_ >= 0) t = m;
    if (t == nullptr) return -1;
    cudaSetDevice(t->dev_);
    // between tokens: the draft block's routes go through the same tiers, so the boundary runs first
    if (!t->fast_boundary(err)) { cudaSetDevice(dev_); return -1; }
    if (!t->fast_mtp(t->pos_ - 1, next_tok, err)) {
        cudaSetDevice(dev_);
        return -1;
    }
    if (!glmfast::wait_event(t->fast_->ev_mtp, "glm mtp completion", err) || !t->fast_->route_ok(err)) {
        cudaSetDevice(dev_);
        return -1;
    }
    const int d = t->fast_->mtp_tok_h[0];
    cudaSetDevice(dev_);
    if (gf::launch_errors() > 0) {
        err = "glm mtp: a kernel launch failed (see stderr)";
        return -1;
    }
    return d;
}

// ---------------------------------------------------------------- pipelined speculative decode (two halves + NextN)
// The two halves of a split take turns on one token, so each GPU idles half the time.  With the NextN block's draft
// for the token after next, the HEAD half runs that draft while the TAIL half finishes the current token: when the
// tail's token equals the draft, the head's work stands and both GPUs stay busy (a token costs the slower half);
// when it differs, the head restores its recurrent states (saved before the speculative token) and reruns the
// position with the real token.  Only the head speculates - the tail, the NextN block and the sampler see confirmed
// tokens only - so the output is exactly what the token-at-a-time decode would produce for the same samples.
bool Glm5Model::spec_ready() const {
    if (fast_ == nullptr || split_next_ == nullptr || getenv("STRATA_GLM_NO_SPEC") != nullptr) return false;
    const Glm5Model* last = this;
    for (const Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        if (m->fast_ == nullptr) return false;
        last = m;
    }
    return last->mtp_il_ >= 0;
}

Glm5Model* Glm5Model::spec_head_last() {
    int n = 0;
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) ++n;
    int h = n / 2;
    if (const char* e = getenv("STRATA_GLM_SPEC_HEAD")) h = std::atoi(e);
    h = std::max(1, std::min(n - 1, h));
    Glm5Model* m = this;
    for (int i = 1; i < h; ++i) m = m->split_next_.get();
    return m;
}

Glm5Model* Glm5Model::spec_tail_last() {
    Glm5Model* m = this;
    while (m->split_next_) m = m->split_next_.get();
    return m;
}

bool Glm5Model::hop_to_next(Glm5Model* B, int64_t p, std::string& err) {
    const size_t hop = (size_t) g_.hc * g_.n_embd * sizeof(float);
    FastState* FA = fast_;
    FastState* FB = B->fast_;
    cudaSetDevice(dev_);
    cudaMemcpyAsync(FA->hop_h, state_, hop, cudaMemcpyDeviceToHost, FA->cs);
    cudaEventRecord(FA->ev_hop, FA->cs);
    cudaSetDevice(B->dev_);
    B->pos_ = p + 1;
    cudaStreamWaitEvent(FB->cs, FA->ev_hop, 0);
    cudaMemcpyAsync(B->state_, FA->hop_h, hop, cudaMemcpyHostToDevice, FB->cs);
    return B->fast_layers(p, true, err);
}

// the head's KDA states (S + conv history of every recurrent layer here: one contiguous run each) <-> kda_bak_
bool Glm5Model::spec_kda_copy(bool restore) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    const size_t run = (size_t) g.d_inner() * g.kda_head_dim + (size_t) 3 * g.d_inner() * (g.d_conv - 1);
    int n_rec = 0;
    for (int il = l0_; il < l1_; ++il) n_rec += g.is_recr(il);
    if (n_rec == 0) return true;
    if (kda_bak_ == nullptr && cudaMalloc(&kda_bak_, (size_t) n_rec * run * sizeof(float)) != cudaSuccess) {
        cudaGetLastError();
        kda_bak_ = nullptr;
        return false;
    }
    int k = 0;
    for (int il = l0_; il < l1_; ++il)
        if (g.is_recr(il)) {
            float* live = state_ + kda_S_[(size_t) il];
            float* bak = kda_bak_ + (size_t) k++ * run;
            cudaMemcpyAsync(restore ? live : bak, restore ? bak : live, run * sizeof(float), cudaMemcpyDeviceToDevice,
                            F->cs);
        }
    return true;
}

// the head half's layers for (p, token); the residual rows go to hop slot p & 1 (spec_ev_hop_ marks them)
bool Glm5Model::spec_head(int64_t p, int32_t token, std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    cudaSetDevice(dev_);
    static const bool sprof = getenv("STRATA_GLM_SPEC_PROF") != nullptr;
    auto t0 = std::chrono::steady_clock::now();
    const auto lap = [&](int i) {
        if (!sprof) return;
        const auto t = std::chrono::steady_clock::now();
        spec_t_[i] += std::chrono::duration<double, std::milli>(t - t0).count();
        t0 = t;
    };
    Glm5Model* hl = spec_head_last();
    // every part of the head group idle first: a part's hop slot and boundary must not race its previous position
    for (Glm5Model* m = split_next_.get(); m != nullptr && m != hl->split_next_.get(); m = m->split_next_.get()) {
        cudaSetDevice(m->dev_);
        if (!glmfast::cuda_ok(cudaStreamSynchronize(m->fast_->cs), "glm spec head part sync", err) ||
            !m->fast_->route_ok(err) || !m->fast_boundary(err)) return false;
    }
    cudaSetDevice(dev_);
    if (!glmfast::cuda_ok(cudaStreamSynchronize(F->cs), "glm spec head sync", err) || !F->route_ok(err)) return false;
    lap(0);
    if (sprof) {
        if (spec_ev_[0] == nullptr)
            for (auto& e : spec_ev_) cudaEventCreate(&e);
        else if (spec_ev_live_) {
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, spec_ev_[0], spec_ev_[1]);
            spec_dev_ms_ += ms;
            ++spec_dev_n_;
        }
        if (spec_ev_live_) {
            float gap = 0.0f;
            cudaEventRecord(spec_ev_[2], F->cs);   // now: the gap since the last token ended
            cudaEventSynchronize(spec_ev_[2]);
            cudaEventElapsedTime(&gap, spec_ev_[1], spec_ev_[2]);
            spec_gap_ms_ += gap;
        }
        cudaEventRecord(spec_ev_[0], F->cs);
    }
    if (F->prof_on) F->collect();
    if (!fast_boundary(err)) return false;
    lap(1);
    const ggml_type_traits* tt = ggml_get_type_traits((ggml_type) pack_emb_type_);
    tt->to_float(pack_emb_src_ + (size_t) token * ggml_row_size((ggml_type) pack_emb_type_, g.n_embd), F->emb_h,
                 g.n_embd);
    if (const float* img = image_row(p)) std::memcpy(F->emb_h, img, (size_t) g.n_embd * sizeof(float));   // an image
    cudaMemcpyAsync(F->emb, F->emb_h, (size_t) g.n_embd * sizeof(float), cudaMemcpyHostToDevice, F->cs);
    embed_in(g, F->emb, state_, F->cs);
    lap(2);
    if (!fast_layers(p, false, err)) return false;
    for (Glm5Model* m = this; m != hl; m = m->split_next_.get())
        if (!m->hop_to_next(m->split_next_.get(), p, err)) return false;
    lap(3);
    ++spec_n_;
    const int sl = (int) (p & 1);
    // the head group's last part hands the residual to the tail group through hl's slots (events on hl's device)
    cudaSetDevice(hl->dev_);
    cudaMemcpyAsync(hl->spec_hop_h_[sl], hl->state_, (size_t) g.hc * g.n_embd * sizeof(float), cudaMemcpyDeviceToHost,
                    hl->fast_->cs);
    cudaEventRecord(hl->spec_ev_hop_[sl], hl->fast_->cs);
    cudaSetDevice(dev_);
    if (sprof) {
        cudaEventRecord(spec_ev_[1], F->cs);
        spec_ev_live_ = true;
    }
    return true;
}

// the tail half (this) for position p: waits on the head's hop slot ON THE DEVICE, then its layers, the head and the
// argmax (tok_h after ev_done)
bool Glm5Model::spec_tail(Glm5Model* head, int64_t p, std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    cudaSetDevice(dev_);
    static const bool sprof = getenv("STRATA_GLM_SPEC_PROF") != nullptr;
    auto t0 = std::chrono::steady_clock::now();
    const auto lap = [&](int i) {
        if (!sprof) return;
        const auto t = std::chrono::steady_clock::now();
        spec_t_[i] += std::chrono::duration<double, std::milli>(t - t0).count();
        t0 = t;
    };
    if (!glmfast::cuda_ok(cudaStreamSynchronize(F->cs), "glm spec tail sync", err)) return false;
    lap(0);
    if (F->prof_on) F->collect();
    if (!fast_boundary(err)) return false;
    lap(1);
    ++spec_n_;
    pos_ = p + 1;
    const int sl = (int) (p & 1);
    if (sprof) {
        if (spec_ev_[0] == nullptr)
            for (auto& e : spec_ev_) cudaEventCreate(&e);
        else if (spec_ev_live_) {
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, spec_ev_[0], spec_ev_[1]);
            spec_dev_ms_ += ms;
            ++spec_dev_n_;
        }
    }
    for (Glm5Model* m = split_next_.get(); m != nullptr; m = m->split_next_.get()) {
        cudaSetDevice(m->dev_);
        if (!glmfast::cuda_ok(cudaStreamSynchronize(m->fast_->cs), "glm spec tail part sync", err) ||
            !m->fast_->route_ok(err) || !m->fast_boundary(err)) return false;
        m->pos_ = p + 1;
    }
    cudaSetDevice(dev_);
    cudaStreamWaitEvent(F->cs, head->spec_ev_hop_[sl], 0);
    if (sprof) cudaEventRecord(spec_ev_[0], F->cs);
    cudaMemcpyAsync(state_, head->spec_hop_h_[sl], (size_t) g.hc * g.n_embd * sizeof(float), cudaMemcpyHostToDevice,
                    F->cs);
    if (!fast_layers(p, true, err)) return false;
    if (split_next_) {
        // a tail GROUP: down its parts; the last one computes the head and the argmax
        Glm5Model* m = this;
        for (; m->split_next_; m = m->split_next_.get())
            if (!m->hop_to_next(m->split_next_.get(), p, err)) return false;
        cudaSetDevice(m->dev_);
        FastState* FL = m->fast_;
        head_in(g, m->state_, m->w_.at("output_norm.weight"), FL->head_x, FL->head_xq, FL->cs);
        const WSlot& owl = m->ws_map_.at("output.weight");
        gf::MvJob ol = {owl.q, FL->head_xq, FL->head_x, m->sc_ + m->sc_logits, nullptr, 1.0f, owl.type, g.n_embd,
                        g.n_vocab};
        if (owl.type == 0) ol.w = owl.f32;
        if (!gf::mv(&ol, 1, FL->cs)) {
            err = "glm spec: output head";
            return false;
        }
        gf::argmax(m->sc_ + m->sc_logits, g.n_vocab, FL->tok, FL->cs);
        cudaMemcpyAsync(FL->tok_h, FL->tok, sizeof(int), cudaMemcpyDeviceToHost, FL->cs);
        cudaEventRecord(FL->ev_done, FL->cs);
        cudaSetDevice(dev_);
        return true;
    }
    head_in(g, state_, w_.at("output_norm.weight"), F->head_x, F->head_xq, F->cs);
    const WSlot& ow = ws_map_.at("output.weight");
    gf::MvJob o = {ow.q, F->head_xq, F->head_x, sc_ + sc_logits, nullptr, 1.0f, ow.type, g.n_embd, g.n_vocab};
    if (ow.type == 0) o.w = ow.f32;
    if (!gf::mv(&o, 1, F->cs)) {
        err = "glm spec: output head";
        return false;
    }
    gf::argmax(sc_ + sc_logits, g.n_vocab, F->tok, F->cs);
    cudaMemcpyAsync(F->tok_h, F->tok, sizeof(int), cudaMemcpyDeviceToHost, F->cs);
    cudaEventRecord(F->ev_done, F->cs);
    if (sprof) {
        cudaEventRecord(spec_ev_[1], F->cs);
        spec_ev_live_ = true;
    }
    return true;
}

bool Glm5Model::decode_spec(strata::kernels::SamplerParams& sp, int64_t max_new, const std::function<bool(int)>& emit,
                            int64_t& produced, std::string& err) {
    produced = 0;
    if (!spec_ready()) {
        err = "glm spec: needs a split with the NextN block";
        return false;
    }
    Glm5Model* hl = spec_head_last();
    Glm5Model* B = hl->split_next_.get();     // the tail group's first part (spec_tail runs from it)
    Glm5Model* TL = spec_tail_last();         // its last: the token, the logits and the draft block live there
    const Glm5Geometry& g = g_;
    if (hl->spec_hop_h_[0] == nullptr) {
        cudaSetDevice(hl->dev_);
        for (int i = 0; i < 2; ++i)
            if (cudaHostAlloc((void**) &hl->spec_hop_h_[i], (size_t) g.hc * g.n_embd * sizeof(float),
                              cudaHostAllocPortable) != cudaSuccess ||
                cudaEventCreateWithFlags(&hl->spec_ev_hop_[i], cudaEventDisableTiming) != cudaSuccess) {
                err = "glm spec: the hop slots did not allocate";
                return false;
            }
        cudaSetDevice(dev_);
    }
    // the head group's recurrent-state backups (each part on its own device and stream)
    const auto kda_all = [&](bool restore) -> bool {
        for (Glm5Model* m = this; m != hl->split_next_.get(); m = m->split_next_.get()) {
            cudaSetDevice(m->dev_);
            if (!m->spec_kda_copy(restore)) return false;
        }
        cudaSetDevice(dev_);
        return true;
    };
    // the tail's token for its last position: the argmax it left, or a sample from its logits
    const auto take = [&](int& y) -> bool {
        if (sp.greedy) {
            cudaSetDevice(TL->dev_);
            if (!glmfast::wait_event(TL->fast_->ev_done, "glm spec token", err)) return false;
            y = TL->fast_->tok_h[0];
        } else {
            y = TL->fast_sample(sp, err);   // the tail group's last part holds the logits
            if (y < 0) return false;
        }
        for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get())
            if (!m->fast_->route_ok(err)) return false;
        y = forced(y);
        sp.counter += 1;
        last_tok_ = y;
        return true;
    };
    const auto draft = [&](int64_t p, int32_t next, int& d) -> bool {
        cudaSetDevice(TL->dev_);
        if (!TL->fast_mtp(p, next, err)) return false;
        if (!glmfast::wait_event(TL->fast_->ev_mtp, "glm spec draft", err) || !TL->fast_->route_ok(err)) return false;
        d = TL->fast_->mtp_tok_h[0];
        return true;
    };
    const auto check = [&]() -> bool {
        if (gf::launch_errors() > 0) {
            err = "glm spec: a kernel launch failed (see stderr)";
            return false;
        }
        return true;
    };
    // the prompt's last forward left position q = pos_ - 1 processed by both halves and its logits on the tail
    int64_t q = pos_ - 1;
    int y = -1;
    if (sp.greedy && last_tok_ >= 0) {
        y = forced(last_tok_);
        sp.counter += 1;
    } else {
        cudaSetDevice(TL->dev_);
        cudaEventRecord(TL->fast_->ev_done, TL->fast_->cs);
        if (!take(y)) return false;
    }
    ++produced;
    if (!emit(y) || produced >= max_new || q + 2 >= max_ctx_) return true;
    int d = -1;
    if (!draft(q, y, d)) return false;
    // the head: position q+1 with the confirmed token, then q+2 with the draft (states saved first)
    if (!spec_head(q + 1, y, err)) return false;
    if (!B->spec_tail(hl, q + 1, err)) return false;
    cudaSetDevice(dev_);
    if (!kda_all(false)) {
        err = "glm spec: the recurrent-state backup did not allocate";
        return false;
    }
    if (!spec_head(q + 2, d, err)) return false;
    bool ok = true;
    // STRATA_GLM_SPEC_PROF=1: host time per phase of a step (debug)
    static const bool sprof = getenv("STRATA_GLM_SPEC_PROF") != nullptr;
    double tp[6] = {0, 0, 0, 0, 0, 0};
    int64_t np = 0;
    auto tnow = std::chrono::steady_clock::now();
    static const char* const lap_names[6] = {"wait tail", "draft", "redo", "enqueue tail", "enqueue head", "emit"};
    const auto lap = [&](int i) {
        if (!sprof) return;
        const auto t = std::chrono::steady_clock::now();
        tp[i] += std::chrono::duration<double, std::milli>(t - tnow).count();
        tnow = t;
#if !defined(STRATA_USE_HIP)
        nvtxRangePop();
        nvtxRangePushA(lap_names[(i + 1) % 6]);
#endif
    };
    // STRATA_GLM_NSYS=<from>,<to>: a profiler capture window over those steps (nsys --capture-range=cudaProfilerApi)
    int64_t nsys_from = -1, nsys_to = -1;
    if (const char* ns = getenv("STRATA_GLM_NSYS")) std::sscanf(ns, "%lld,%lld", (long long*) &nsys_from, (long long*) &nsys_to);
#if !defined(STRATA_USE_HIP)
    if (sprof) nvtxRangePushA("emit");
#endif
    for (;;) {
        if (np == nsys_from) cudaProfilerStart();
        if (np == nsys_to) cudaProfilerStop();
        // (1) the tail's token for position q+1 -> y (the truth for position q+2)
        int y2 = -1;
        lap(5);
        if (!take(y2)) { ok = false; break; }
        lap(0);
        ++produced;
        const bool more = emit(y2) && produced < max_new && q + 3 < max_ctx_;
        if (!more) break;
        // (2) the draft for q+3 from (h_{q+1}, y2)
        int d2 = -1;
        if (!draft(q + 1, y2, d2)) { ok = false; break; }
        lap(1);
        // (3) the head's speculative position q+2 stands iff its draft was y2; else restore and redo it
        ++spec_steps_;
        if (d == y2) {
            ++spec_hits_;
        } else {
            kda_all(true);
            if (!spec_head(q + 2, y2, err)) { ok = false; break; }
        }
        lap(2);
        // (4) the tail's next position goes in first (it waits for the head's hop on the device), then the head's
        //     next speculative position (its boundary waits for the head's current one)
        if (!B->spec_tail(hl, q + 2, err)) { ok = false; break; }
        lap(3);
        kda_all(false);
        if (!spec_head(q + 3, d2, err)) { ok = false; break; }
        lap(4);
        ++np;
        if (!check()) { ok = false; break; }
        ++q;
        d = d2;
    }
    // drain: both devices idle; the head may stand one or two positions past the tail - a request after this one
    // restores its snapshot or resets, so nothing else reads that state
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        cudaSetDevice(m->dev_);
        std::string drain_err;
        if (!glmfast::cuda_ok(cudaStreamSynchronize(m->fast_->cs), "glm spec drain", drain_err) ||
            !m->fast_->route_ok(drain_err)) {
            if (ok) err = drain_err;
            ok = false;
        }
    }
    cudaSetDevice(dev_);
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) m->pos_ = TL->pos_;
    // the head's recurrent states back to the tail's position (the backup after q+1): the whole model then stands at
    // pos_ exactly, so a snapshot taken now (the server's PAUSE) continues the sequence; a reset or a restore after
    // this overwrites them anyway
    if (ok && !kda_all(true)) ok = false;
    if (sprof && np > 0) {
        std::fprintf(stderr, "glm spec prof (%lld steps, ms/step): wait tail token %.2f | draft %.2f | redo %.2f | "
                             "enqueue tail %.2f | enqueue head %.2f | emit %.2f\n", (long long) np, tp[0] / np, tp[1] / np,
                     tp[2] / np, tp[3] / np, tp[4] / np, tp[5] / np);
        for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get())
            if (m->spec_n_ > 0)
                std::fprintf(stderr, "glm spec prof CUDA%d (%lld calls, ms/call): sync %.3f | boundary %.3f | embed %.3f | "
                                     "enqueue layers %.3f | DEVICE token %.2f ms, gap before %.2f ms\n", m->dev_,
                             (long long) m->spec_n_, m->spec_t_[0] / m->spec_n_, m->spec_t_[1] / m->spec_n_,
                             m->spec_t_[2] / m->spec_n_, m->spec_t_[3] / m->spec_n_,
                             m->spec_dev_ms_ / (double) std::max<int64_t>(1, m->spec_dev_n_),
                             m->spec_gap_ms_ / (double) std::max<int64_t>(1, m->spec_dev_n_));
    }
    if (ok && !check()) ok = false;
    return ok;
}

// ---------------------------------------------------------------- conversation reuse
// The sequence state that cannot be rebuilt from the position alone is the KDA layers' recurrent state and conv
// history (one contiguous run per KDA layer in state_); the DSA caches are append-only, so restoring the position
// is enough for them (entries past it are rewritten as the new tokens arrive).
std::vector<int> Glm5Model::dsa_layers() const {
    std::vector<int> r;
    if (mtp_il_ >= 0) r.push_back(mtp_il_);
    for (int il = l0_; il < l1_; ++il)
        if (g_.is_dsa(il)) r.push_back(il);
    return r;
}

// the open pool's ik / ig rows at snap_pos_ (pool start .. + kpool - 2: contiguous in the ring, which is a multiple
// of kpool) <-> snap_pool_; on fast_->cs
void Glm5Model::snap_pool_copy(bool restore) {
    const Glm5Geometry& g = g_;
    const std::vector<int> ls = dsa_layers();
    const int64_t rows = (int64_t) (g.idx_kpool - 1) * g.idx_key;   // floats per cache
    if (ls.empty() || rows <= 0 || snap_pos_ < 0) return;
    if (snap_pool_ == nullptr &&
        cudaMalloc(&snap_pool_, (size_t) ls.size() * 2 * rows * sizeof(float)) != cudaSuccess) {
        cudaGetLastError();
        snap_pool_ = nullptr;
        return;
    }
    const int64_t at = (int64_t) g.idx_key * ((snap_pos_ - snap_pos_ % g.idx_kpool) % ik_ring_);
    for (size_t i = 0; i < ls.size(); ++i) {
        float* ring[2] = {state_ + dsa_ik_[(size_t) ls[i]] + at, state_ + dsa_ig_[(size_t) ls[i]] + at};
        for (int c = 0; c < 2; ++c) {
            float* keep = snap_pool_ + (i * 2 + (size_t) c) * rows;
            cudaMemcpyAsync(restore ? ring[c] : keep, restore ? keep : ring[c], (size_t) rows * sizeof(float),
                            cudaMemcpyDeviceToDevice, fast_->cs);
        }
    }
}

std::vector<std::pair<int64_t, int64_t>> Glm5Model::d4_snap_runs() const {
    std::vector<std::pair<int64_t, int64_t>> r;
    if (!g_.dsv4()) return r;
    for (int il = l0_; il < l1_; ++il) {
        r.push_back({d4_swa_[(size_t) il], (int64_t) g_.n_swa * g_.head_dim / 2});
        if (d4_st_[(size_t) il] >= 0) r.push_back({d4_st_[(size_t) il], d4_st_floats(il)});
        if (d4_ist_[(size_t) il] >= 0) r.push_back({d4_ist_[(size_t) il], (int64_t) 2 * 8 * 2 * g_.idx_key});
    }
    return r;
}
int64_t Glm5Model::d4_snap_floats() const {
    int64_t n = 0;
    for (const auto& r : d4_snap_runs()) n += r.second;
    return n;
}
std::vector<std::pair<int64_t, int64_t>> Glm5Model::slot_d4_runs(int64_t n) const {
    std::vector<std::pair<int64_t, int64_t>> r;
    if (!g_.dsv4()) return r;
    for (int il = l0_; il < l1_; ++il) {
        const int ratio = g_.ratio(il);
        if (ratio == 0) continue;
        r.push_back({d4_comp_[(size_t) il], (n / ratio * g_.head_dim + 1) / 2});
        if (ratio == 4) r.push_back({d4_icomp_[(size_t) il], (n / 4 * g_.idx_key + 1) / 2});
    }
    return r;
}

bool Glm5Model::snapshot_save() {
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        if (m->fast_ == nullptr) return false;
        if (m->g_.dsv4()) {   // deepseek4: the rings and windows, back to back
            const int64_t nf = m->d4_snap_floats();
            cudaSetDevice(m->dev_);
            if (m->snap_ == nullptr && cudaMalloc(&m->snap_, (size_t) nf * sizeof(float)) != cudaSuccess) {
                cudaGetLastError();
                m->snap_ = nullptr;
                cudaSetDevice(dev_);
                return false;
            }
            int64_t at = 0;
            for (const auto& r : m->d4_snap_runs()) {
                cudaMemcpyAsync(m->snap_ + at, m->state_ + r.first, (size_t) r.second * sizeof(float),
                                cudaMemcpyDeviceToDevice, m->fast_->cs);
                at += r.second;
            }
            m->snap_pos_ = pos_;
            cudaStreamSynchronize(m->fast_->cs);
            continue;
        }
        const Glm5Geometry& g = m->g_;
        const size_t run = (size_t) g.d_inner() * g.kda_head_dim + (size_t) 3 * g.d_inner() * (g.d_conv - 1);
        int n_rec = 0;
        for (int il = m->l0_; il < m->l1_; ++il) n_rec += g.is_recr(il);
        cudaSetDevice(m->dev_);
        if (m->snap_ == nullptr && n_rec > 0 &&
            cudaMalloc(&m->snap_, (size_t) n_rec * run * sizeof(float)) != cudaSuccess) {
            cudaGetLastError();
            m->snap_ = nullptr;
            cudaSetDevice(dev_);
            return false;
        }
        int k = 0;
        for (int il = m->l0_; il < m->l1_; ++il)
            if (g.is_recr(il))
                cudaMemcpyAsync(m->snap_ + (size_t) k++ * run, m->state_ + m->kda_S_[(size_t) il], run * sizeof(float),
                                cudaMemcpyDeviceToDevice, m->fast_->cs);
        m->snap_pos_ = pos_;
        m->snap_pool_copy(false);
        cudaStreamSynchronize(m->fast_->cs);
        if (m->snap_pool_ == nullptr && !m->dsa_layers().empty()) {
            cudaSetDevice(dev_);
            return false;
        }
    }
    cudaSetDevice(dev_);
    return true;
}

bool Glm5Model::snapshot_restore() {
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get())
        if (m->fast_ == nullptr || m->snap_pos_ < 0) return false;
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        const Glm5Geometry& g = m->g_;
        if (g.dsv4()) {
            cudaSetDevice(m->dev_);
            int64_t at = 0;
            for (const auto& r : m->d4_snap_runs()) {
                cudaMemcpyAsync(m->state_ + r.first, m->snap_ + at, (size_t) r.second * sizeof(float),
                                cudaMemcpyDeviceToDevice, m->fast_->cs);
                at += r.second;
            }
            cudaStreamSynchronize(m->fast_->cs);
            m->pos_ = m->snap_pos_;
            continue;
        }
        const size_t run = (size_t) g.d_inner() * g.kda_head_dim + (size_t) 3 * g.d_inner() * (g.d_conv - 1);
        cudaSetDevice(m->dev_);
        int k = 0;
        for (int il = m->l0_; il < m->l1_; ++il)
            if (g.is_recr(il))
                cudaMemcpyAsync(m->state_ + m->kda_S_[(size_t) il], m->snap_ + (size_t) k++ * run, run * sizeof(float),
                                cudaMemcpyDeviceToDevice, m->fast_->cs);
        m->snap_pool_copy(true);
        cudaStreamSynchronize(m->fast_->cs);
        m->pos_ = m->snap_pos_;
    }
    cudaSetDevice(dev_);
    return true;
}

// ---------------------------------------------------------------- conversation slots
// A conversation set aside while others run: per half, the snapshot's KDA states (snap_, taken just before its prompt's
// last token) and every DSA cache's rows up to that position - the latents, the indexer's two key rows and the pools,
// the NextN block's too.  The file: a header (magic, halves, positions, per half its layer range and run sizes), then
// per half the KDA snapshot and the DSA runs in that order.
namespace {
constexpr uint64_t kSlotMagic = 0x3254534C4159414Dull;   // "MAYALST2" (v2: the ik / ig ring's open pool only)

// (state_ offset, floats) of each DSA cache run a slot of n positions holds on one half, in file order
std::vector<std::pair<int64_t, int64_t>> slot_dsa_runs(const Glm5Geometry& g, bool fast_mode, bool lat_q8, int l0, int l1,
                                                       int mtp_il,
                                                       const std::vector<int64_t>& lat, const std::vector<int64_t>& ik,
                                                       const std::vector<int64_t>& ig, const std::vector<int64_t>& pool,
                                                       int64_t n) {
    std::vector<std::pair<int64_t, int64_t>> r;
    // FP16 latents take half a float each, INT8 records lat8_rec_bytes / 4
    const int64_t lat_pp = lat_q8      ? (int64_t) gf::lat8_rec_bytes(g.kv_lora) / 4
                           : fast_mode ? g.kv_lora / 2
                                       : g.kv_lora;
    const int64_t pools = (n + g.idx_kpool - 1) / g.idx_kpool;
    const auto dsa = [&](size_t il) {
        r.push_back({lat[il], n * lat_pp});
        r.push_back({pool[il], pools * g.idx_key});   // (ik / ig: the open pool's rows, in snap_pool_)
        (void) ik;
        (void) ig;
    };
    if (mtp_il >= 0) dsa((size_t) mtp_il);
    for (int il = l0; il < l1; ++il)
        if (g.is_dsa(il)) dsa((size_t) il);
    return r;
}
}  // namespace

// hy_v3: (state_ offset, floats) of each K / V cache run a slot of n positions holds on this part - per attention layer
// (the NextN block's first) the K rows, the V rows, and with an INT8 cache their scales
std::vector<std::pair<int64_t, int64_t>> Glm5Model::slot_gqa_runs(int64_t n) const {
    // the records of positions [0, n): region a's in this arena, region b's in the pool's tail (an offset from state_
    // as well: the save and the load add it to state_)
    std::vector<std::pair<int64_t, int64_t>> r;
    if (!g_.gqa()) return r;
    if (kvs_ != nullptr) {   // streamed: every layer's host records (an offset from state_ as well)
        for (int il : kvs_->layers)
            r.push_back({(int64_t) ((const float*) kvs_->lay[(size_t) il].host - state_), n * (int64_t) kvs_->slice / 4});
        return r;
    }
    const int64_t p0 = kv_b_ != nullptr ? kv_p0_ : max_ctx_, first = gqa_kv_[(size_t) l0_];
    r.push_back({first, std::min(n, p0) * kv_rec_ / 4});
    if (kv_b_ != nullptr && n > p0) r.push_back({(int64_t) ((const float*) kv_lo(n) - state_), (n - p0) * kv_rec_ / 4});
    return r;
}

uint64_t Glm5Model::slot_save(const std::string& path, std::string& err) {
    std::vector<Glm5Model*> halves;
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) {
        if (m->fast_ == nullptr || m->snap_pos_ < 0 || m->snap_pos_ != snap_pos_) {
            err = "no snapshot to save";
            return 0;
        }
        halves.push_back(m);
    }
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) {
        err = "cannot write " + path;
        return 0;
    }
    const int64_t n = snap_pos_;
    const auto put = [&](const void* p, size_t bytes) { f.write((const char*) p, (std::streamsize) bytes); };
    const uint32_t nh = (uint32_t) halves.size();
    put(&kSlotMagic, 8);
    put(&nh, 4);
    put(&n, 8);
    std::vector<float> buf;
    double t_sync = 0, t_copy = 0, t_write = 0;
    for (Glm5Model* m : halves) {
        const Glm5Geometry& g = m->g_;
        const int64_t run = (int64_t) g.d_inner() * g.kda_head_dim + (int64_t) 3 * g.d_inner() * (g.d_conv - 1);
        int32_t n_rec = 0;
        for (int il = m->l0_; il < m->l1_; ++il) n_rec += g.is_recr(il);
        auto runs = slot_dsa_runs(g, m->fast_mode_, m->lat_q8_, m->l0_, m->l1_, m->mtp_il_, m->dsa_lat_, m->dsa_ik_,
                                  m->dsa_ig_, m->dsa_pool_, n);
        const auto gruns = m->slot_gqa_runs(n);   // hy_v3's K / V rows
        runs.insert(runs.end(), gruns.begin(), gruns.end());
        const auto druns = m->slot_d4_runs(n);    // deepseek4's pooled rows
        runs.insert(runs.end(), druns.begin(), druns.end());
        const int32_t hdr[4] = {m->l0_, m->l1_, n_rec, (int32_t) runs.size()};
        put(hdr, sizeof(hdr));
        put(&run, 8);
        for (const auto& r : runs) put(&r.second, 8);
        cudaSetDevice(m->dev_);
        auto tq = std::chrono::steady_clock::now();
        cudaStreamSynchronize(m->fast_->cs);
        t_sync += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tq).count();
        std::vector<std::pair<const float*, int64_t>> src;
        if (n_rec > 0) src.push_back({m->snap_, (int64_t) n_rec * run});
        if (g.dsv4()) src.push_back({m->snap_, m->d4_snap_floats()});
        const int64_t pool_rows = (int64_t) m->dsa_layers().size() * 2 * (g.idx_kpool - 1) * g.idx_key;
        if (pool_rows > 0) src.push_back({m->snap_pool_, pool_rows});
        for (const auto& r : runs) src.push_back({m->state_ + r.first, r.second});
        for (const auto& s : src) {
            constexpr int64_t kChunk = (int64_t) 16 << 20;   // floats
            for (int64_t at = 0; at < s.second; at += kChunk) {
                const int64_t c = std::min(kChunk, s.second - at);
                buf.resize((size_t) c);
                tq = std::chrono::steady_clock::now();
                if (cudaMemcpy(buf.data(), s.first + at, (size_t) c * sizeof(float), cudaMemcpyDefault) !=
                    cudaSuccess) {
                    cudaGetLastError();
                    cudaSetDevice(dev_);
                    err = "the state did not copy back";
                    return 0;
                }
                const auto tw = std::chrono::steady_clock::now();
                t_copy += std::chrono::duration<double, std::milli>(tw - tq).count();
                put(buf.data(), (size_t) c * sizeof(float));
                t_write += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tw).count();
            }
        }
    }
    cudaSetDevice(dev_);
    if (getenv("STRATA_GLM_SLOT_TIMING"))
        std::fprintf(stderr, "glm slots: save %lld positions - sync %.0f ms, copy %.0f ms, write %.0f ms\n",
                     (long long) n, t_sync, t_copy, t_write);
    f.flush();
    if (!f) {
        err = "writing " + path + " failed (disk full?)";
        return 0;
    }
    return (uint64_t) f.tellp();
}

bool Glm5Model::slot_load(const std::string& path, int64_t n_pos, std::string& err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        err = "cannot read " + path;
        return false;
    }
    const auto get = [&](void* p, size_t bytes) { return (bool) f.read((char*) p, (std::streamsize) bytes); };
    uint64_t magic = 0;
    uint32_t nh = 0;
    int64_t n = 0;
    std::vector<Glm5Model*> halves;
    for (Glm5Model* m = this; m != nullptr; m = m->split_next_.get()) halves.push_back(m);
    if (!get(&magic, 8) || !get(&nh, 4) || !get(&n, 8) || magic != kSlotMagic || nh != halves.size() || n != n_pos ||
        n <= 0 || n > max_ctx_) {
        err = "not a slot of this model";
        return false;
    }
    std::vector<float> buf;
    for (Glm5Model* m : halves) {
        if (m->fast_ == nullptr) {
            err = "no fast path";
            return false;
        }
        const Glm5Geometry& g = m->g_;
        const int64_t run = (int64_t) g.d_inner() * g.kda_head_dim + (int64_t) 3 * g.d_inner() * (g.d_conv - 1);
        int32_t n_rec = 0;
        for (int il = m->l0_; il < m->l1_; ++il) n_rec += g.is_recr(il);
        auto runs = slot_dsa_runs(g, m->fast_mode_, m->lat_q8_, m->l0_, m->l1_, m->mtp_il_, m->dsa_lat_, m->dsa_ik_,
                                  m->dsa_ig_, m->dsa_pool_, n);
        const auto gruns = m->slot_gqa_runs(n);   // hy_v3's K / V rows
        runs.insert(runs.end(), gruns.begin(), gruns.end());
        const auto druns = m->slot_d4_runs(n);    // deepseek4's pooled rows
        runs.insert(runs.end(), druns.begin(), druns.end());
        int32_t hdr[4] = {};
        int64_t frun = 0;
        if (!get(hdr, sizeof(hdr)) || !get(&frun, 8) || hdr[0] != m->l0_ || hdr[1] != m->l1_ || hdr[2] != n_rec ||
            hdr[3] != (int32_t) runs.size() || frun != run) {
            err = "the slot's layout differs from this model's";
            return false;
        }
        for (const auto& r : runs) {
            int64_t fl = 0;
            if (!get(&fl, 8) || fl != r.second) {
                err = "the slot's layout differs from this model's";
                return false;
            }
        }
        cudaSetDevice(m->dev_);
        if (!m->kv_ensure(n, err)) {
            cudaSetDevice(dev_);
            return false;
        }
        cudaStreamSynchronize(m->fast_->cs);
        if (m->snap_ == nullptr && n_rec > 0 &&
            cudaMalloc(&m->snap_, (size_t) n_rec * run * sizeof(float)) != cudaSuccess) {
            cudaGetLastError();
            m->snap_ = nullptr;
            cudaSetDevice(dev_);
            err = "the snapshot did not allocate";
            return false;
        }
        if (g.dsv4() && m->snap_ == nullptr &&
            cudaMalloc(&m->snap_, (size_t) m->d4_snap_floats() * sizeof(float)) != cudaSuccess) {
            cudaGetLastError();
            m->snap_ = nullptr;
            cudaSetDevice(dev_);
            err = "the snapshot did not allocate";
            return false;
        }
        std::vector<std::pair<float*, int64_t>> dst;
        if (n_rec > 0) dst.push_back({m->snap_, (int64_t) n_rec * run});
        if (g.dsv4()) dst.push_back({m->snap_, m->d4_snap_floats()});
        const int64_t pool_rows = (int64_t) m->dsa_layers().size() * 2 * (g.idx_kpool - 1) * g.idx_key;
        if (pool_rows > 0) {
            if (m->snap_pool_ == nullptr && cudaMalloc(&m->snap_pool_, (size_t) pool_rows * sizeof(float)) != cudaSuccess) {
                cudaGetLastError();
                m->snap_pool_ = nullptr;
                cudaSetDevice(dev_);
                err = "the snapshot did not allocate";
                return false;
            }
            dst.push_back({m->snap_pool_, pool_rows});
        }
        for (const auto& r : runs) dst.push_back({m->state_ + r.first, r.second});
        for (const auto& d : dst) {
            constexpr int64_t kChunk = (int64_t) 16 << 20;
            for (int64_t at = 0; at < d.second; at += kChunk) {
                const int64_t c = std::min(kChunk, d.second - at);
                buf.resize((size_t) c);
                if (!get(buf.data(), (size_t) c * sizeof(float)) ||
                    cudaMemcpy(d.first + at, buf.data(), (size_t) c * sizeof(float), cudaMemcpyDefault) !=
                        cudaSuccess) {
                    cudaGetLastError();
                    cudaSetDevice(dev_);
                    err = "the slot did not read back";
                    return false;
                }
            }
        }
        m->snap_pos_ = n;
        m->kv_stream_reload(n);   // (streamed: the pages' summaries of the records read back, nothing cached)
    }
    cudaSetDevice(dev_);
    // the KDA states go from the snapshot into the live state, as snapshot_restore does (it also sets the position)
    if (!snapshot_restore()) {
        err = "the snapshot did not restore";
        return false;
    }
    return true;
}

bool Glm5Model::forward_fast(const std::vector<int32_t>& tokens, std::vector<float>& logits_out, std::string& err) {
    FastState* F = fast_;
    for (int32_t t : tokens) {
        const auto t0 = std::chrono::steady_clock::now();
        if (!fast_token(t, err)) return false;
        F->ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        ++F->tokens;
    }
    if (host_logits_) {
        Glm5Model* tail = this;
        while (tail->split_next_) tail = tail->split_next_.get();
        cudaSetDevice(tail->dev_);
        logits_out.resize((size_t) g_.n_vocab);
        if (cudaMemcpy(logits_out.data(), tail->sc_ + tail->sc_logits, (size_t) g_.n_vocab * sizeof(float),
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
            err = "glm fast: logits copy";
            return false;
        }
        cudaSetDevice(dev_);
    }
    return true;
}


// ---------------------------------------------------------------- deepseek4's MTP block (STRATA_DSV4_MTP)
int Glm5Model::d4_mtp_exp_type() {
    const char* v = getenv("STRATA_DSV4_MTP_EXPERTS");
    if (v == nullptr || !v[0]) return -1;
    const std::string t(v);
    return t == "q2_k" ? GGML_TYPE_Q2_K : t == "q3_k" ? GGML_TYPE_Q3_K : t == "q4_k" ? -1 : -1;
}
// ds4.c metal_graph_eval_mtp_draft_from_hc: the input streams R[s] = e_proj(enorm(emb(token))) + h_proj(hnorm(R_trunk[s]))
// through one DSv4 layer (an own raw window, no compressor: layer 1's rope), then the mHC head on the block's own
// hc_head / norm and the trunk's output head.  Its window holds only the block's own rows (from the first decode
// step on: the prompt never went through it), at positions counted from that step (the rope is relative).
struct Glm5Model::D4Mtp {
    std::unique_ptr<strata::GgufFile> f;
    const float *enorm, *hnorm, *norm, *attn_norm, *ffn_norm, *q_a_norm, *kv_a_norm, *sinks, *ha_scale, *ha_base,
        *hf_scale, *hf_base, *hh_scale, *hh_base, *router;
    const uint16_t *ha_fn, *hf_fn, *hh_fn;
    const void *q_a, *q_b, *kv, *o_a, *o_b, *e_proj, *h_proj, *sh_g, *sh_u, *sh_d;
    std::vector<float> bias_h;
    uint8_t* exp_dev = nullptr;              // all 256 experts' [gate | up | down] (null: staged per token)
    uint8_t* stage_dev = nullptr;            // the staged experts (k of them)
    uint8_t* stage_h = nullptr;
    size_t gu_bytes = 0, d_bytes = 0, blob = 0;
    int gt = 0, dt = 0;          // the experts' types (gate/up, down)
    const uint8_t *src_g = nullptr, *src_u = nullptr, *src_d = nullptr;
    uint16_t* swa = nullptr;
    float *R, *R2, *ep, *hp, *ex, *xs, *glog, *eg, *eu, *ey, *shg, *shu, *shy;
    void *exq, *xsq, *ehq, *shq;
    float* glog_h = nullptr;
    float* logits = nullptr;
    int* tok = nullptr;
    int* tok_h = nullptr;
    int64_t last_pos = -2, vpos = 0;
};

namespace {
// the softmax probability of the largest of n logits (one block of 1024): 1 / sum exp(l - max)
__global__ void __launch_bounds__(1024) d4_top1_prob_kernel(const float* __restrict__ l, int n, float* __restrict__ out) {
    __shared__ float sh[32];
    float m = -INFINITY;
    for (int i = threadIdx.x; i < n; i += 1024) m = fmaxf(m, l[i]);
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = m;
    __syncthreads();
    m = sh[0];
    for (int i = 1; i < 32; ++i) m = fmaxf(m, sh[i]);
    __syncthreads();
    float z = 0.0f;
    for (int i = threadIdx.x; i < n; i += 1024) z += __expf(l[i] - m);
    for (int o = 16; o > 0; o >>= 1) z += __shfl_xor_sync(0xffffffffu, z, o);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = z;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.0f;
        for (int i = 0; i < 32; ++i) t += sh[i];
        *out = 1.0f / t;
    }
}
}  // namespace

bool Glm5Model::d4_mtp_load(const std::string& path, std::string& err) {
    const Glm5Geometry& g = g_;
    FastState* F = fast_;
    auto* M = new D4Mtp;
    try {
        M->f = std::make_unique<strata::GgufFile>(path);
    } catch (const std::exception& e) {
        err = std::string("dsv4 mtp: ") + e.what();
        return false;
    }
    size_t total = 0;
    const auto dev = [&](const std::string& n, int want, const strata::TensorInfo** out = nullptr) -> const void* {
        const strata::TensorInfo* t = M->f->find("mtp.0." + n);
        if (t == nullptr || (want >= 0 && (int) t->type != want)) {
            err = "dsv4 mtp: " + n + (t ? " has an unexpected type" : " missing");
            return nullptr;
        }
        if (out) *out = t;
        const size_t bytes = ggml_row_size((ggml_type) t->type, (int64_t) t->shape[0]) * (t->elements() / t->shape[0]);
        void* d = nullptr;
        if (cudaMalloc(&d, bytes) != cudaSuccess ||
            cudaMemcpy(d, M->f->tensor_data(*t), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            err = "dsv4 mtp: no VRAM for " + n;
            return nullptr;
        }
        total += bytes;
        return d;
    };
    const auto f16 = [&](const std::string& n) -> const uint16_t* {
        const strata::TensorInfo* t = M->f->find("mtp.0." + n);
        if (t == nullptr || t->type != GGML_TYPE_F32) {
            err = "dsv4 mtp: " + n + " missing or not F32";
            return nullptr;
        }
        std::vector<ggml_fp16_t> h((size_t) t->elements());
        ggml_fp32_to_fp16_row((const float*) M->f->tensor_data(*t), h.data(), (int64_t) h.size());
        void* d = nullptr;
        if (cudaMalloc(&d, h.size() * 2) != cudaSuccess ||
            cudaMemcpy(d, h.data(), h.size() * 2, cudaMemcpyHostToDevice) != cudaSuccess) {
            err = "dsv4 mtp: no VRAM for " + n;
            return nullptr;
        }
        total += h.size() * 2;
        return (const uint16_t*) d;
    };
    const int F32 = GGML_TYPE_F32, Q8 = GGML_TYPE_Q8_0;
#define D4F(v, n) if ((M->v = (const float*) dev(n, F32)) == nullptr) return false
#define D4Q(v, n) if ((M->v = dev(n, Q8)) == nullptr) return false
#define D4H(v, n) if ((M->v = f16(n)) == nullptr) return false
    D4F(enorm, "enorm.weight"); D4F(hnorm, "hnorm.weight"); D4F(norm, "norm.weight");
    D4F(attn_norm, "attn_norm.weight"); D4F(ffn_norm, "ffn_norm.weight"); D4F(q_a_norm, "attn_q_a_norm.weight");
    D4F(kv_a_norm, "attn_kv_a_norm.weight"); D4F(sinks, "attn_sinks.weight");
    D4F(ha_scale, "hc_attn_scale.weight"); D4F(ha_base, "hc_attn_base.weight");
    D4F(hf_scale, "hc_ffn_scale.weight"); D4F(hf_base, "hc_ffn_base.weight");
    D4F(hh_scale, "hc_head_scale.weight"); D4F(hh_base, "hc_head_base.weight"); D4F(router, "ffn_gate_inp.weight");
    D4H(ha_fn, "hc_attn_fn.weight"); D4H(hf_fn, "hc_ffn_fn.weight"); D4H(hh_fn, "hc_head_fn.weight");
    D4Q(q_a, "attn_q_a.weight"); D4Q(q_b, "attn_q_b.weight"); D4Q(kv, "attn_kv.weight");
    D4Q(o_a, "attn_output_a.weight"); D4Q(o_b, "attn_output_b.weight"); D4Q(e_proj, "e_proj.weight");
    D4Q(h_proj, "h_proj.weight"); D4Q(sh_g, "ffn_gate_shexp.weight"); D4Q(sh_u, "ffn_up_shexp.weight");
    D4Q(sh_d, "ffn_down_shexp.weight");
#undef D4F
#undef D4Q
#undef D4H
    {
        const strata::TensorInfo* b = M->f->find("mtp.0.exp_probs_b.bias");
        if (b == nullptr) { err = "dsv4 mtp: exp_probs_b.bias missing"; return false; }
        const float* bp = (const float*) M->f->tensor_data(*b);
        M->bias_h.assign(bp, bp + b->elements());
    }
    // the routed experts (Q4_K): all in VRAM when they fit, else the picked ones staged a token
    const strata::TensorInfo *tg = M->f->find("mtp.0.ffn_gate_exps.weight"), *tu = M->f->find("mtp.0.ffn_up_exps.weight"),
                             *td = M->f->find("mtp.0.ffn_down_exps.weight");
    if (!tg || !tu || !td || tg->type != tu->type || !gf::mv_supported((int) tg->type) || !gf::mv_supported((int) td->type)) {
        err = "dsv4 mtp: routed experts missing or of an unsupported type";
        return false;
    }
    // STRATA_DSV4_MTP_EXPERTS=q3_k|q2_k: the experts requantized (from Q4_K) to leave the trunk's pool more VRAM
    const int rq = d4_mtp_exp_type();
    M->gt = rq >= 0 ? rq : (int) tg->type;
    M->dt = rq >= 0 ? rq : (int) td->type;
    const size_t src_gu = gf::row_bytes((int) tg->type, g.n_embd) * g.n_ff_exp,
                 src_d = gf::row_bytes((int) td->type, g.n_ff_exp) * g.n_embd;
    M->gu_bytes = gf::row_bytes(M->gt, g.n_embd) * g.n_ff_exp;
    M->d_bytes = gf::row_bytes(M->dt, g.n_ff_exp) * g.n_embd;
    M->blob = 2 * M->gu_bytes + M->d_bytes;
    M->src_g = M->f->tensor_data(*tg);
    M->src_u = M->f->tensor_data(*tu);
    M->src_d = M->f->tensor_data(*td);
    std::vector<uint8_t> rq_buf;
    if (rq >= 0) {
        // [gate | up | down] per expert in the new type, on all cores (ggml's quantizers)
        const auto t0 = std::chrono::steady_clock::now();
        rq_buf.resize(M->blob * (size_t) g.n_expert);
        std::atomic<int> next{0};
        const auto work = [&] {
            std::vector<float> f((size_t) g.n_ff_exp * g.n_embd);
            for (int e; (e = next++) < g.n_expert;) {
                uint8_t* dst = rq_buf.data() + (size_t) e * M->blob;
                const ggml_type_traits* sg = ggml_get_type_traits((ggml_type) tg->type);
                const ggml_type_traits* sd = ggml_get_type_traits((ggml_type) td->type);
                sg->to_float(M->src_g + (size_t) e * src_gu, f.data(), (int64_t) g.n_ff_exp * g.n_embd);
                ggml_quantize_chunk((ggml_type) M->gt, f.data(), dst, 0, g.n_ff_exp, g.n_embd, nullptr);
                sg->to_float(M->src_u + (size_t) e * src_gu, f.data(), (int64_t) g.n_ff_exp * g.n_embd);
                ggml_quantize_chunk((ggml_type) M->gt, f.data(), dst + M->gu_bytes, 0, g.n_ff_exp, g.n_embd, nullptr);
                sd->to_float(M->src_d + (size_t) e * src_d, f.data(), (int64_t) g.n_ff_exp * g.n_embd);
                ggml_quantize_chunk((ggml_type) M->dt, f.data(), dst + 2 * M->gu_bytes, 0, g.n_embd, g.n_ff_exp, nullptr);
            }
        };
        std::vector<std::thread> th;
        const int nt = std::max(1, (int) std::thread::hardware_concurrency());
        for (int i = 0; i < nt; ++i) th.emplace_back(work);
        for (auto& t : th) t.join();
        std::fprintf(stderr, "dsv4 mtp: experts requantized to %s in %.1f s\n", ggml_type_name((ggml_type) rq),
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    const int K = g.n_exp_used;
    if (getenv("STRATA_DSV4_MTP_STAGE") == nullptr &&
        cudaMalloc(&M->exp_dev, M->blob * (size_t) g.n_expert) == cudaSuccess) {
        if (rq >= 0) cudaMemcpy(M->exp_dev, rq_buf.data(), rq_buf.size(), cudaMemcpyHostToDevice);
        for (int e = 0; e < g.n_expert && rq < 0; ++e) {
            uint8_t* d = M->exp_dev + (size_t) e * M->blob;
            cudaMemcpy(d, M->src_g + (size_t) e * M->gu_bytes, M->gu_bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d + M->gu_bytes, M->src_u + (size_t) e * M->gu_bytes, M->gu_bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d + 2 * M->gu_bytes, M->src_d + (size_t) e * M->d_bytes, M->d_bytes, cudaMemcpyHostToDevice);
        }
        total += M->blob * (size_t) g.n_expert;
    } else {
        cudaGetLastError();
        M->exp_dev = nullptr;
        if (rq >= 0) { err = "dsv4 mtp: no VRAM for the requantized experts"; return false; }
        if (cudaMalloc(&M->stage_dev, M->blob * (size_t) K) != cudaSuccess ||
            cudaHostAlloc((void**) &M->stage_h, M->blob * (size_t) K, cudaHostAllocDefault) != cudaSuccess) {
            err = "dsv4 mtp: no room for the staged experts";
            return false;
        }
        total += M->blob * (size_t) K;
    }
    // activations
    const int E = g.n_embd, FF = g.n_ff_exp;
    const auto q8 = [](size_t n) { return n / 32 * 36; };
    const size_t fl = (size_t) 4 * E * 3 + E + E + E + g.n_expert + (size_t) K * FF * 2 + (size_t) K * E + 2 * (size_t) FF + E;
    float* a = nullptr;
    if (cudaMalloc(&a, fl * sizeof(float)) != cudaSuccess || cudaMalloc(&M->swa, (size_t) g.n_swa * 512 * 2) != cudaSuccess ||
        cudaMalloc(&M->exq, q8(E)) != cudaSuccess || cudaMalloc(&M->xsq, q8(4 * (size_t) E)) != cudaSuccess ||
        cudaMalloc(&M->ehq, q8((size_t) K * FF)) != cudaSuccess || cudaMalloc(&M->shq, q8(FF)) != cudaSuccess ||
        cudaMalloc(&M->tok, 2 * sizeof(int)) != cudaSuccess || cudaMalloc(&M->logits, (size_t) g.n_vocab * sizeof(float)) != cudaSuccess ||
        cudaHostAlloc((void**) &M->tok_h, 2 * sizeof(int), cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc((void**) &M->glog_h, g.n_expert * sizeof(float), cudaHostAllocDefault) != cudaSuccess) {
        err = "dsv4 mtp: activation buffers did not allocate";
        return false;
    }
    M->R = a; a += 4 * (size_t) E;
    M->R2 = a; a += 4 * (size_t) E;
    M->hp = a; a += 4 * (size_t) E;
    M->ep = a; a += E;
    M->ex = a; a += E;
    M->xs = a; a += E;
    M->glog = a; a += g.n_expert;
    M->eg = a; a += (size_t) K * FF;
    M->eu = a; a += (size_t) K * FF;
    M->ey = a; a += (size_t) K * E;
    M->shg = a; a += FF;
    M->shu = a; a += FF;
    M->shy = a;
    (void) F;
    d4mtp_ = M;
    d4mtp_resident_ = M->exp_dev != nullptr;
    std::fprintf(stderr, "dsv4 mtp: %s - %.2f GB of VRAM, experts %s\n", path.c_str(), total / 1e9,
                 M->exp_dev ? "resident" : "staged a token");
    return true;
}

// one draft: (token at position pos, the trunk's streams at pos - 1 in state_) -> the block's argmax at pos + 1
bool Glm5Model::d4_mtp_draft(int32_t token, int64_t pos, int& draft, std::string& err, bool kv_only) {
    D4Mtp* M = d4mtp_;
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    const int E = g.n_embd, HD = g.head_dim, K = g.n_exp_used, FF = g.n_ff_exp;
    const auto t_in = std::chrono::steady_clock::now();
    if (pos != M->last_pos + 1) M->vpos = 0;   // a new run (a prompt in between): the block's window starts over
    M->last_pos = pos;
    const int vp = (int) M->vpos++;
    // ---- the input streams
    const ggml_type_traits* tt = ggml_get_type_traits((ggml_type) pack_emb_type_);
    tt->to_float(pack_emb_src_ + (size_t) token * ggml_row_size((ggml_type) pack_emb_type_, E), F->emb_h, E);
    cudaMemcpyAsync(F->emb, F->emb_h, (size_t) E * sizeof(float), cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(M->ex, F->emb, (size_t) E * sizeof(float), cudaMemcpyDeviceToDevice, s);
    gf::rms_q8(M->ex, nullptr, M->enorm, g.norm_eps, E, M->xs, M->exq, s);
    gf::MvJob j[gf::kMaxMvJobs];
    int nj = 0;
    j[nj++] = {M->e_proj, M->exq, nullptr, M->ep, nullptr, 1.0f, GGML_TYPE_Q8_0, E, E};
    for (int h = 0; h < 4; ++h) {
        float* r = M->R2 + (size_t) h * E;
        cudaMemcpyAsync(r, state_ + (size_t) h * E, (size_t) E * sizeof(float), cudaMemcpyDeviceToDevice, s);
        gf::rms_q8(r, nullptr, M->hnorm, g.norm_eps, E, M->xs, (uint8_t*) M->xsq + (size_t) h * (E / 32) * 36, s);
        j[nj++] = {M->h_proj, (uint8_t*) M->xsq + (size_t) h * (E / 32) * 36, nullptr, M->hp + (size_t) h * E, nullptr,
                   1.0f, GGML_TYPE_Q8_0, E, E};
    }
    if (!gf::mv(j, nj, s)) { err = "dsv4 mtp: e/h proj"; return false; }
    gf::dsv4_mtp_input(M->ep, M->hp, M->R, E, s);
    // ---- the attention half
    gf::HcArgs h;
    h.block_out = nullptr;
    h.R_old = M->R;
    h.R_new = M->R2;
    h.post_in = F->post;
    h.comb_in = F->comb;
    h.pre = F->pre;
    h.post = F->post;
    h.comb = F->comb;
    h.w_fn = M->ha_fn;
    h.w_f16 = true;
    h.w_scale = M->ha_scale;
    h.w_base = M->ha_base;
    h.norm_w = M->attn_norm;
    h.norm_eps = g.norm_eps;
    h.hc_eps = g.hc_eps;
    h.iters = g.sinkhorn_iters;
    h.n_embd = E;
    h.x = F->x;
    h.xq = F->xq;
    h.part = F->part;
    h.counter = F->counter;
    gf::hc(h, s);
    // STRATA_DSV4_MTP_KVONLY=0: an accepted round's window row through the whole block (the old dummy draft)
    static const bool kv_only_on = !getenv("STRATA_DSV4_MTP_KVONLY") || atoi(getenv("STRATA_DSV4_MTP_KVONLY")) != 0;
    kv_only = kv_only && kv_only_on;
    nj = 0;
    if (!kv_only) j[nj++] = {M->q_a, F->xq, nullptr, F->d4_qr, nullptr, 1.0f, GGML_TYPE_Q8_0, E, g.q_lora};
    j[nj++] = {M->kv, F->xq, nullptr, F->d4_kv, nullptr, 1.0f, GGML_TYPE_Q8_0, E, HD};
    if (!gf::mv(j, nj, s)) { err = "dsv4 mtp: q_a / kv"; return false; }
    if (kv_only) {   // the row's K/V into the window: no q, attention, FFN or head
        gf::Dsv4Layer L0;
        L0.swa = M->swa;
        L0.ratio = 0;
        const gf::Dsv4KvArgs ka0{F->d4_kv, M->kv_a_norm, nullptr, nullptr, nullptr, nullptr,
                                 nullptr, nullptr, nullptr, nullptr, g.norm_eps, g.n_swa};
        gf::dsv4_kv_update(ka0, L0, F->d4_rope[1], vp, s);
        draft = -1;
        mtp_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_in).count();
        return true;
    }
    gf::dsv4_qr_norm(F->d4_qr, M->q_a_norm, g.norm_eps, g.q_lora, F->d4_qrn, F->d4_qrn_q8, s);
    j[0] = {M->q_b, F->d4_qrn_q8, nullptr, F->d4_q, nullptr, 1.0f, GGML_TYPE_Q8_0, g.q_lora, g.n_head * HD};
    if (!gf::mv(j, 1, s)) { err = "dsv4 mtp: q_b"; return false; }
    gf::Dsv4Layer L;
    L.swa = M->swa;
    L.ratio = 0;
    const gf::Dsv4Rope& rope = F->d4_rope[1];
    const gf::Dsv4KvArgs ka{F->d4_kv, M->kv_a_norm, nullptr, nullptr, nullptr, nullptr,
                            nullptr, nullptr, nullptr, nullptr, g.norm_eps, g.n_swa};
    gf::dsv4_kv_update(ka, L, rope, vp, s);
    gf::dsv4_attn(F->d4_q, M->sinks, L, g.n_swa, vp, nullptr, 0, rope, g.norm_eps, g.n_head, F->d4_heads, s);
    gf::quantize_q8_1(F->d4_heads, F->d4_heads_q8, g.n_head * HD, s);
    const int GD = g.n_head / g.o_groups * HD;
    const size_t wrow = gf::row_bytes(GGML_TYPE_Q8_0, GD);
    for (int gr = 0; gr < g.o_groups; ++gr)
        j[gr] = {(const uint8_t*) M->o_a + (size_t) gr * g.o_lora * wrow,
                 (const uint8_t*) F->d4_heads_q8 + (size_t) gr * (GD / 32) * 36, nullptr, F->d4_low + (size_t) gr * g.o_lora,
                 nullptr, 1.0f, GGML_TYPE_Q8_0, GD, g.o_lora};
    if (!gf::mv(j, g.o_groups, s)) { err = "dsv4 mtp: output a"; return false; }
    gf::quantize_q8_1(F->d4_low, F->d4_low_q8, g.o_groups * g.o_lora, s);
    j[0] = {M->o_b, F->d4_low_q8, nullptr, F->mixer, nullptr, 1.0f, GGML_TYPE_Q8_0, g.o_groups * g.o_lora, E};
    if (!gf::mv(j, 1, s)) { err = "dsv4 mtp: output b"; return false; }
    // ---- the FFN half: R2 = post x mixer + comb . R, its normed read in x / xq
    h.block_out = F->mixer;
    h.R_old = M->R;
    h.R_new = M->R2;
    h.w_fn = M->hf_fn;
    h.w_scale = M->hf_scale;
    h.w_base = M->hf_base;
    h.norm_w = M->ffn_norm;
    gf::hc(h, s);
    nj = 0;
    j[nj++] = {M->router, nullptr, F->x, M->glog, nullptr, 1.0f, gf::kTypeF32, E, g.n_expert};
    j[nj++] = {M->sh_g, F->xq, nullptr, M->shg, nullptr, 1.0f, GGML_TYPE_Q8_0, E, FF};
    j[nj++] = {M->sh_u, F->xq, nullptr, M->shu, nullptr, 1.0f, GGML_TYPE_Q8_0, E, FF};
    if (!gf::mv(j, nj, s)) { err = "dsv4 mtp: router / shared"; return false; }
    gf::swiglu_q8(M->shg, M->shu, g.swiglu_shexp, FF, M->shq, s);
    j[0] = {M->sh_d, M->shq, nullptr, M->shy, nullptr, 1.0f, GGML_TYPE_Q8_0, FF, E};
    if (!gf::mv(j, 1, s)) { err = "dsv4 mtp: shared down"; return false; }
    cudaMemcpyAsync(M->glog_h, M->glog, g.n_expert * sizeof(float), cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);
    // the route (ds4: probs = sqrt(softplus(logit)), picks on probs + bias, weights the picks' probs normalised x scale)
    std::vector<float> pr((size_t) g.n_expert), key((size_t) g.n_expert);
    for (int e = 0; e < g.n_expert; ++e) {
        const float l = M->glog_h[e];
        pr[(size_t) e] = std::sqrt(l > 20.0f ? l : std::log1p(std::exp(l)));
        key[(size_t) e] = pr[(size_t) e] + M->bias_h[(size_t) e];
    }
    std::vector<int> id((size_t) g.n_expert);
    for (int e = 0; e < g.n_expert; ++e) id[(size_t) e] = e;
    std::partial_sort(id.begin(), id.begin() + K, id.end(), [&](int x, int y) { return key[(size_t) x] > key[(size_t) y]; });
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) sum += pr[(size_t) id[(size_t) k]];
    gf::Dsv4MtpW wt{};
    for (int k = 0; k < K; ++k) wt.w[k] = pr[(size_t) id[(size_t) k]] / sum * g.w_scale;
    const int gt = M->gt, dt = M->dt;
    const uint8_t* blob[8];
    if (M->exp_dev) {
        for (int k = 0; k < K; ++k) blob[k] = M->exp_dev + (size_t) id[(size_t) k] * M->blob;
    } else {
        for (int k = 0; k < K; ++k) {
            const int e = id[(size_t) k];
            uint8_t* hb = M->stage_h + (size_t) k * M->blob;
            std::memcpy(hb, M->src_g + (size_t) e * M->gu_bytes, M->gu_bytes);
            std::memcpy(hb + M->gu_bytes, M->src_u + (size_t) e * M->gu_bytes, M->gu_bytes);
            std::memcpy(hb + 2 * M->gu_bytes, M->src_d + (size_t) e * M->d_bytes, M->d_bytes);
            blob[k] = M->stage_dev + (size_t) k * M->blob;
        }
        cudaMemcpyAsync(M->stage_dev, M->stage_h, M->blob * (size_t) K, cudaMemcpyHostToDevice, s);
    }
    for (int k0 = 0; k0 < K; k0 += 4) {
        nj = 0;
        for (int k = k0; k < std::min(K, k0 + 4); ++k) {
            j[nj++] = {blob[k], F->xq, nullptr, M->eg + (size_t) k * FF, nullptr, 1.0f, gt, E, FF};
            j[nj++] = {blob[k] + M->gu_bytes, F->xq, nullptr, M->eu + (size_t) k * FF, nullptr, 1.0f, gt, E, FF};
        }
        if (!gf::mv(j, nj, s)) { err = "dsv4 mtp: expert gate / up"; return false; }
    }
    for (int k = 0; k < K; ++k)
        gf::swiglu_q8(M->eg + (size_t) k * FF, M->eu + (size_t) k * FF, g.swiglu_exp, FF,
                      (uint8_t*) M->ehq + (size_t) k * (FF / 32) * 36, s);
    for (int k = 0; k < K; ++k)
        j[k] = {blob[k] + 2 * M->gu_bytes, (uint8_t*) M->ehq + (size_t) k * (FF / 32) * 36, nullptr, M->ey + (size_t) k * E,
                nullptr, 1.0f, dt, FF, E};
    if (!gf::mv(j, K, s)) { err = "dsv4 mtp: expert down"; return false; }
    gf::dsv4_mtp_combine(M->ey, wt, K, M->shy, F->ffn, E, s);
    gf::hc_post(F->ffn, M->R2, F->post, F->comb, E, M->R, s);
    // ---- the head
    gf::dsv4_hc_head(M->R, M->hh_fn, M->hh_scale, M->hh_base, g.hc_eps, M->norm, g.norm_eps, E, F->head_x, F->head_xq, s);
    const WSlot& ow = ws_map_.at("output.weight");
    gf::MvJob o = {ow.q, F->head_xq, F->head_x, M->logits, nullptr, 1.0f, ow.type, E, g.n_vocab};
    if (ow.type == 0) o.w = ow.f32;
    if (!gf::mv(&o, 1, s)) { err = "dsv4 mtp: output head"; return false; }
    gf::argmax(M->logits, g.n_vocab, M->tok, s);
    d4_top1_prob_kernel<<<1, 1024, 0, s>>>(M->logits, g.n_vocab, (float*) (M->tok + 1));
    cudaMemcpyAsync(M->tok_h, M->tok, 2 * sizeof(int), cudaMemcpyDeviceToHost, s);
    if (cudaStreamSynchronize(s) != cudaSuccess || cudaGetLastError() != cudaSuccess) {
        err = "dsv4 mtp: the draft failed";
        return false;
    }
    draft = M->tok_h[0];
    std::memcpy(&d4_mtp_conf_, M->tok_h + 1, sizeof(float));
    mtp_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_in).count();
    return true;
}

// ---------------------------------------------------------------- deepseek4: the MTP speculative decode
// The second row's buffers: swapped with the FastState's own for a per-row step (the kernels read F's fields)
struct Glm5Model::D4Spec {
    float *x, *mixer, *ffn, *pre, *post, *comb, *head_x;
    void *xq, *head_xq;
    float *d4_qr, *d4_qrn, *d4_q, *d4_kv, *d4_ckv, *d4_csc, *d4_ickv, *d4_icsc, *d4_iq, *d4_iw, *d4_score, *d4_heads,
        *d4_low;
    void *d4_qrn_q8, *d4_heads_q8, *d4_low_q8;
    int* d4_sel;
    float* R = nullptr;          // row B's streams: [Rc | Ro], 2 x hc x n_embd
    float* Rb = nullptr;         // row B's final streams
    float* logits = nullptr;     // row B's
    float* win = nullptr;        // the compressor windows before row B, per layer
    struct Run { int il; int64_t off, n, at; };
    std::vector<Run> runs;       // every layer's compressor windows: state_ offset, floats, offset in win
    int* tok = nullptr;          // 2: the rows' argmax
    int* tok_h = nullptr;
};

template<class FS, class SB>
static void d4_swap_rows(FS* F, SB* B) {
    std::swap(F->x, B->x); std::swap(F->mixer, B->mixer); std::swap(F->ffn, B->ffn); std::swap(F->pre, B->pre);
    std::swap(F->post, B->post); std::swap(F->comb, B->comb); std::swap(F->head_x, B->head_x);
    std::swap(F->xq, B->xq); std::swap(F->head_xq, B->head_xq);
    std::swap(F->d4_qr, B->d4_qr); std::swap(F->d4_qrn, B->d4_qrn); std::swap(F->d4_q, B->d4_q);
    std::swap(F->d4_kv, B->d4_kv); std::swap(F->d4_ckv, B->d4_ckv); std::swap(F->d4_csc, B->d4_csc);
    std::swap(F->d4_ickv, B->d4_ickv); std::swap(F->d4_icsc, B->d4_icsc); std::swap(F->d4_iq, B->d4_iq);
    std::swap(F->d4_iw, B->d4_iw); std::swap(F->d4_score, B->d4_score); std::swap(F->d4_heads, B->d4_heads);
    std::swap(F->d4_low, B->d4_low); std::swap(F->d4_qrn_q8, B->d4_qrn_q8); std::swap(F->d4_heads_q8, B->d4_heads_q8);
    std::swap(F->d4_low_q8, B->d4_low_q8); std::swap(F->d4_sel, B->d4_sel);
}

bool Glm5Model::d4_spec_alloc(std::string& err) {
    FastState* F = fast_;
    const Glm5Geometry& g = g_;
    auto* B = new D4Spec;
    const size_t E = (size_t) g.n_embd;
    const auto q8f = [](size_t n) { return (n / 32 * 36 + 3) / 4; };
    const size_t QL = (size_t) g.q_lora, QH = (size_t) g.n_head * g.head_dim, LO = (size_t) g.o_groups * g.o_lora,
                 IQ = (size_t) g.idx_heads * g.idx_key, NC = (size_t) std::max<int64_t>(1, max_ctx_ / 4 + 1);
    int64_t wn = 0;
    for (int il = l0_; il < l1_; ++il) {
        if (d4_st_[(size_t) il] >= 0) { B->runs.push_back({il, d4_st_[(size_t) il], d4_st_floats(il), wn}); wn += d4_st_floats(il); }
        if (d4_ist_[(size_t) il] >= 0) {
            const int64_t n = (int64_t) 2 * 8 * 2 * g.idx_key;
            B->runs.push_back({il, d4_ist_[(size_t) il], n, wn});
            wn += n;
        }
    }
    std::vector<size_t> sz = {E, E, E, 8, 8, 16, E, q8f(E), q8f(E), QL, QL, q8f(QL), QH, (size_t) g.head_dim,
                              2 * (size_t) g.head_dim, 2 * (size_t) g.head_dim, 2 * (size_t) g.idx_key,
                              2 * (size_t) g.idx_key, IQ, (size_t) g.idx_heads, NC, QH, q8f(QH), LO, q8f(LO),
                              (size_t) g.idx_top_k, 4 * g.hc * E, (size_t) g.hc * E, (size_t) g.n_vocab, (size_t) wn, 16};
    size_t tot = 0;
    for (size_t v : sz) tot += (v + 15) / 16 * 16;
    float* at = nullptr;
    if (cudaMalloc(&at, tot * sizeof(float)) != cudaSuccess ||
        cudaHostAlloc((void**) &B->tok_h, 2 * sizeof(int), cudaHostAllocDefault) != cudaSuccess) {
        err = "dsv4 spec: the second row's buffers did not allocate";
        return false;
    }
    size_t k = 0;
    const auto take = [&]() { float* r = at; at += (sz[k++] + 15) / 16 * 16; return r; };
    B->x = take(); B->mixer = take(); B->ffn = take(); B->pre = take(); B->post = take(); B->comb = take();
    B->head_x = take(); B->xq = take(); B->head_xq = take();
    B->d4_qr = take(); B->d4_qrn = take(); B->d4_qrn_q8 = take(); B->d4_q = take(); B->d4_kv = take();
    B->d4_ckv = take(); B->d4_csc = take(); B->d4_ickv = take(); B->d4_icsc = take(); B->d4_iq = take();
    B->d4_iw = take(); B->d4_score = take(); B->d4_heads = take(); B->d4_heads_q8 = take(); B->d4_low = take();
    B->d4_low_q8 = take(); B->d4_sel = (int*) take(); B->R = take(); B->Rb = take(); B->logits = take();
    B->win = take(); B->tok = (int*) take();
    (void) F;
    d4spec_ = B;
    std::fprintf(stderr, "dsv4 mtp spec: on (greedy decode), %.1f MB for the second row\n", tot * 4.0 / 1e6);
    return true;
}

bool Glm5Model::d4_spec_ready() const {
    return d4mtp_ != nullptr && d4spec_ != nullptr && split_next_ == nullptr && fast_ != nullptr;
}

// rows A (token a at p = pos_) and B (token d at p + 1) through every layer; ya / yb: the trunk's argmax after each.
// Leaves A's streams in state_, B's in d4spec_->Rb, pos_ unchanged, B's windows saved in d4spec_->win.
bool Glm5Model::d4_verify2(int32_t a, int32_t d, int& ya, int& yb, std::string& err) {
    FastState* F = fast_;
    D4Spec* B = d4spec_;
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    const int E = g.n_embd, HD = g.head_dim;
    const int64_t p = pos_;
    if (!fast_boundary(err)) return false;
    const ggml_type_traits* tt = ggml_get_type_traits((ggml_type) pack_emb_type_);
    const size_t rb = ggml_row_size((ggml_type) pack_emb_type_, E);
    tt->to_float(pack_emb_src_ + (size_t) a * rb, F->emb_h, E);
    cudaMemcpyAsync(F->emb, F->emb_h, (size_t) E * sizeof(float), cudaMemcpyHostToDevice, s);
    gf::embed_streams(F->emb, state_, E, s);
    cudaStreamSynchronize(s);
    tt->to_float(pack_emb_src_ + (size_t) d * rb, F->emb_h, E);
    cudaMemcpyAsync(F->emb, F->emb_h, (size_t) E * sizeof(float), cudaMemcpyHostToDevice, s);
    gf::embed_streams(F->emb, B->R, E, s);
    float* Rc[2] = {state_, B->R};
    float* Ro[2] = {state_ + (int64_t) g.hc * E, B->R + (int64_t) g.hc * E};
    bool pf_pending = false;
    if (F->prof_on) F->mark("start");
    const auto hc_args = [&](int r, const FastState::Layer& Ly, bool ffn_side, bool first) {
        gf::HcArgs h;
        h.block_out = ffn_side ? F->mixer : first ? nullptr : F->ffn;
        h.R_old = Rc[r];
        h.R_new = Ro[r];
        h.post_in = F->post;
        h.comb_in = F->comb;
        h.pre = F->pre;
        h.post = F->post;
        h.comb = F->comb;
        h.w_fn = ffn_side ? Ly.hc_ffn_fn : Ly.hc_attn_fn;
        h.w_f16 = true;
        h.w_scale = ffn_side ? Ly.hc_ffn_scale : Ly.hc_attn_scale;
        h.w_base = ffn_side ? Ly.hc_ffn_base : Ly.hc_attn_base;
        h.norm_w = ffn_side ? Ly.ffn_norm : Ly.attn_norm;
        h.norm_eps = g.norm_eps;
        h.hc_eps = g.hc_eps;
        h.iters = g.sinkhorn_iters;
        h.n_embd = E;
        h.x = F->x;
        h.xq = F->xq;
        h.part = F->part;
        h.counter = F->counter;
        gf::hc(h, s);
    };
    // row r's step on F's fields (row 1: B's buffers swapped in)
    const auto on_row = [&](int r, const std::function<void()>& f) {
        if (r) d4_swap_rows(F, B);
        f();
        if (r) d4_swap_rows(F, B);
    };
    // the default order: the rows half a layer apart, so a row's attention runs on the GPU while the CPU lane computes
    // the other row's experts - route A, A's device experts, B's attention, A's CPU answer, route B, B's device
    // experts, A's next attention, B's CPU answer, ... (one route's device work in flight at a time, as in decode).
    // STRATA_DSV4_SPEC_JOINT=1: both rows' attention together (the dense weights read once), then each row's MoE.
    static const bool joint = getenv("STRATA_DSV4_SPEC_JOINT") != nullptr && std::atoi(getenv("STRATA_DSV4_SPEC_JOINT")) != 0;
    if (!joint) {
        const auto attn = [&](int r, int il) -> bool {
            const auto& Ly = F->L[(size_t) il];
            const bool first = il == l0_;
            bool ok = true;
            on_row(r, [&] {
                hc_args(r, Ly, false, first);
                if (!first) std::swap(Rc[r], Ro[r]);
                if (r == 1)   // row B's windows can be rolled back
                    for (const auto& u : B->runs)
                        if (u.il == il)
                            cudaMemcpyAsync(B->win + u.at, state_ + u.off, (size_t) u.n * sizeof(float),
                                            cudaMemcpyDeviceToDevice, s);
                ok = fast_dsv4(il, p + r, err);
                hc_args(r, Ly, true, first);
                std::swap(Rc[r], Ro[r]);
            });
            if (F->prof_on) F->mark(r ? "v_attn_b" : "v_attn_a");
            return ok;
        };
        const auto moe = [&](int r, int il) -> bool {
            bool ok = true;
            cur_tok_ = r ? d : a;
            on_row(r, [&] { ok = fast_moe(il, pf_pending, err, false); });
            return ok;
        };
        const auto lane_wait = [&](int r, int il) {
            if (!fast_moe_lane(il)) return;
            on_row(r, [&] { gf::moe_cpu_wait(F->md, E, F->ffn, s); });
            if (F->prof_on) F->mark("v_cpu_wait");
        };
        if (!attn(0, l0_)) return false;
        for (int il = l0_; il < l1_; ++il) {
            if (!moe(0, il)) return false;
            if (!attn(1, il)) return false;
            lane_wait(0, il);
            if (!moe(1, il)) return false;
            if (il + 1 < l1_ && !attn(0, il + 1)) return false;
            lane_wait(1, il);
        }
    }
    for (int il = l0_; il < l1_ && joint; ++il) {
        const auto& Ly = F->L[(size_t) il];
        const bool first = il == l0_;
        for (int r = 0; r < 2; ++r) {
            on_row(r, [&] { hc_args(r, Ly, false, first); });
            if (!first) std::swap(Rc[r], Ro[r]);
        }
        if (F->prof_on) F->mark("v_hc_attn");
        // ---- the attention: the projections of both rows in one pass over the weights
        const int rt = g.ratio(il), W = rt == 4 ? 2 * HD : HD, IW = 2 * g.idx_key;
        gf::MvJob j[2][gf::kMaxMvJobs];
        int nj = 0;
        for (int r = 0; r < 2; ++r)
            on_row(r, [&] {
                int n = 0;
                j[r][n++] = {Ly.q_a.q, F->xq, nullptr, F->d4_qr, nullptr, 1.0f, Ly.q_a.type, E, g.q_lora};
                j[r][n++] = {Ly.kv_a.q, F->xq, nullptr, F->d4_kv, nullptr, 1.0f, Ly.kv_a.type, E, HD};
                if (rt > 0) {
                    j[r][n++] = {Ly.c_kv, nullptr, F->x, F->d4_ckv, nullptr, 1.0f, gf::kTypeF16, E, W};
                    j[r][n++] = {Ly.c_gate, nullptr, F->x, F->d4_csc, nullptr, 1.0f, gf::kTypeF16, E, W};
                }
                if (rt == 4) {
                    j[r][n++] = {Ly.ic_kv, nullptr, F->x, F->d4_ickv, nullptr, 1.0f, gf::kTypeF16, E, IW};
                    j[r][n++] = {Ly.ic_gate, nullptr, F->x, F->d4_icsc, nullptr, 1.0f, gf::kTypeF16, E, IW};
                    j[r][n++] = {Ly.idx_proj, nullptr, F->x, F->d4_iw, nullptr, 1.0f, gf::kTypeF16, E, g.idx_heads};
                }
                nj = n;
            });
        if (!gf::mv2(j[0], j[1], nj, s)) { err = "dsv4 spec: projections"; return false; }
        if (F->prof_on) F->mark("v_proj_mv");
        for (int r = 0; r < 2; ++r)
            on_row(r, [&] {
                gf::dsv4_qr_norm(F->d4_qr, Ly.q_a_norm, g.norm_eps, g.q_lora, F->d4_qrn, F->d4_qrn_q8, s);
                int n = 0;
                j[r][n++] = {Ly.q_b.q, F->d4_qrn_q8, nullptr, F->d4_q, nullptr, 1.0f, Ly.q_b.type, g.q_lora, g.n_head * HD};
                if (rt == 4)
                    j[r][n++] = {Ly.idx_q_b, nullptr, F->d4_qrn, F->d4_iq, nullptr, 1.0f, gf::kTypeF16, g.q_lora,
                                 g.idx_heads * g.idx_key};
                nj = n;
            });
        if (!gf::mv2(j[0], j[1], nj, s)) { err = "dsv4 spec: q"; return false; }
        if (F->prof_on) F->mark("v_q_mv");
        const gf::Dsv4Layer L = d4_layer(il);
        const gf::Dsv4Rope& rope = F->d4_rope[(size_t) il];
        for (int r = 0; r < 2; ++r) {
            if (r == 1)   // row B's windows can be rolled back
                for (const auto& u : B->runs)
                    if (u.il == il)
                        cudaMemcpyAsync(B->win + u.at, state_ + u.off, (size_t) u.n * sizeof(float),
                                        cudaMemcpyDeviceToDevice, s);
            on_row(r, [&] {
                const int64_t pr = p + r;
                const gf::Dsv4KvArgs ka{F->d4_kv, Ly.kv_a_norm, F->d4_ckv, F->d4_csc, Ly.c_ape, Ly.c_norm,
                                        F->d4_ickv, F->d4_icsc, Ly.ic_ape, Ly.ic_norm, g.norm_eps, g.n_swa};
                gf::dsv4_kv_update(ka, L, rope, (int) pr, s);
                const int n_comp = rt > 0 ? (int) ((pr + 1) / rt) : 0;
                const int* sel = nullptr;
                int n_sel = n_comp;
                if (rt == 4 && n_comp > g.idx_top_k) {
                    gf::dsv4_indexer(F->d4_iq, F->d4_iw, g.idx_heads, L, n_comp, rope, (int) pr, g.idx_top_k,
                                     F->d4_score, F->d4_sel, s);
                    sel = F->d4_sel;
                    n_sel = g.idx_top_k;
                }
                gf::dsv4_attn(F->d4_q, Ly.sinks, L, g.n_swa, (int) pr, sel, n_sel, rope, g.norm_eps, g.n_head,
                              F->d4_heads, s);
                gf::quantize_q8_1(F->d4_heads, F->d4_heads_q8, g.n_head * HD, s);
            });
        }
        if (F->prof_on) F->mark("v_attn");
        const int GD = g.n_head / g.o_groups * HD;
        const size_t wrow = gf::row_bytes(Ly.o_a.type, GD);
        for (int r = 0; r < 2; ++r)
            on_row(r, [&] {
                for (int gr = 0; gr < g.o_groups; ++gr)
                    j[r][gr] = {(const uint8_t*) Ly.o_a.q + (size_t) gr * g.o_lora * wrow,
                                (const uint8_t*) F->d4_heads_q8 + (size_t) gr * (GD / 32) * 36, nullptr,
                                F->d4_low + (size_t) gr * g.o_lora, nullptr, 1.0f, Ly.o_a.type, GD, g.o_lora};
            });
        if (!gf::mv2(j[0], j[1], g.o_groups, s)) { err = "dsv4 spec: output a"; return false; }
        for (int r = 0; r < 2; ++r)
            on_row(r, [&] {
                gf::quantize_q8_1(F->d4_low, F->d4_low_q8, g.o_groups * g.o_lora, s);
                j[r][0] = {Ly.o_b.q, F->d4_low_q8, nullptr, F->mixer, nullptr, 1.0f, Ly.o_b.type,
                           g.o_groups * g.o_lora, E};
            });
        if (!gf::mv2(j[0], j[1], 1, s)) { err = "dsv4 spec: output b"; return false; }
        if (F->prof_on) F->mark("v_out_mv");
        // ---- the FFN side: per row (the expert machinery is one row's)
        for (int r = 0; r < 2; ++r) {
            on_row(r, [&] { hc_args(r, Ly, true, first); });
            std::swap(Rc[r], Ro[r]);
        }
        if (F->prof_on) F->mark("v_hc_ffn");
        bool ok = true;
        for (int r = 0; r < 2 && ok; ++r) {
            cur_tok_ = r ? d : a;
            on_row(r, [&] {
                if (!Ly.moe) {
                    gf::MvJob dj[2];
                    dj[0] = {Ly.ffn_gate.q, F->xq, nullptr, F->dg, nullptr, 1.0f, Ly.ffn_gate.type, E, g.n_ff_dense};
                    dj[1] = {Ly.ffn_up.q, F->xq, nullptr, F->du, nullptr, 1.0f, Ly.ffn_up.type, E, g.n_ff_dense};
                    ok = gf::mv(dj, 2, s);
                    gf::swiglu_q8(F->dg, F->du, g.swiglu_shexp, g.n_ff_dense, F->dhq, s);
                    gf::MvJob dn = {Ly.ffn_down.q, F->dhq, nullptr, F->ffn, nullptr, 1.0f, Ly.ffn_down.type,
                                    g.n_ff_dense, E};
                    ok = ok && gf::mv(&dn, 1, s);
                } else {
                    ok = fast_moe(il, pf_pending, err);
                }
            });
        }
        if (!ok) { if (err.empty()) err = "dsv4 spec: ffn"; return false; }
    }
    // the last layer's write halves, then the heads
    gf::hc_post(F->ffn, Rc[0], F->post, F->comb, E, Ro[0], s);
    if (Ro[0] != state_) cudaMemcpyAsync(state_, Ro[0], (size_t) g.hc * E * sizeof(float), cudaMemcpyDeviceToDevice, s);
    gf::hc_post(B->ffn, Rc[1], B->post, B->comb, E, B->Rb, s);
    const uint16_t* hfn = w16_.at("output_hc_fn.weight");
    const float *hsc = w_.at("output_hc_scale.weight"), *hba = w_.at("output_hc_base.weight"),
                *onorm = w_.at("output_norm.weight");
    gf::dsv4_hc_head(state_, hfn, hsc, hba, g.hc_eps, onorm, g.norm_eps, E, F->head_x, F->head_xq, s);
    gf::dsv4_hc_head(B->Rb, hfn, hsc, hba, g.hc_eps, onorm, g.norm_eps, E, B->head_x, B->head_xq, s);
    const WSlot& ow = ws_map_.at("output.weight");
    gf::MvJob oa = {ow.q, F->head_xq, F->head_x, sc_ + sc_logits, nullptr, 1.0f, ow.type, E, g.n_vocab};
    gf::MvJob ob = {ow.q, B->head_xq, B->head_x, B->logits, nullptr, 1.0f, ow.type, E, g.n_vocab};
    if (ow.type == 0) oa.w = ob.w = ow.f32;
    if (!gf::mv2(&oa, &ob, 1, s)) { err = "dsv4 spec: output head"; return false; }
    gf::argmax(sc_ + sc_logits, g.n_vocab, B->tok, s);
    gf::argmax(B->logits, g.n_vocab, B->tok + 1, s);
    if (F->prof_on) F->mark("v_head");
    cudaMemcpyAsync(B->tok_h, B->tok, 2 * sizeof(int), cudaMemcpyDeviceToHost, s);
    cudaEventRecord(F->ev_done, s);
    cudaError_t done;
    while ((done = cudaEventQuery(F->ev_done)) == cudaErrorNotReady) std::this_thread::yield();
    if (!glmfast::cuda_ok(done, "dsv4 spec round", err)) return false;
    if (!F->route_ok(err)) return false;
    if (gf::launch_errors() > 0) { err = "dsv4 spec: kernel launch(es) failed"; return false; }
    ya = B->tok_h[0];
    yb = B->tok_h[1];
    if (F->prof_on) F->collect();
    return true;
}

bool Glm5Model::d4_decode_spec(strata::kernels::SamplerParams& sp, int64_t max_new, const std::function<bool(int)>& emit,
                               int64_t& produced, std::string& err) {
    FastState* F = fast_;
    D4Spec* B = d4spec_;
    const Glm5Geometry& g = g_;
    cudaStream_t s = F->cs;
    const size_t hcb = (size_t) g.hc * g.n_embd * sizeof(float);
    produced = 0;
    const auto rollback = [&] {   // row B's windows back
        for (const auto& u : B->runs)
            cudaMemcpyAsync(state_ + u.off, B->win + u.at, (size_t) u.n * sizeof(float), cudaMemcpyDeviceToDevice, s);
    };
    int a = sample_token(sp, err);
    if (a < 0) return false;
    a = forced(a);
    sp.counter += 1;
    ++produced;
    if (!emit(a)) return true;
    int dft = -1;
    if (!d4_mtp_draft(a, pos_, dft, err)) return false;
    const auto t0 = std::chrono::steady_clock::now();
    const int64_t tok0 = F->tokens;
    const uint64_t r0 = d4_rounds_;
    double verify_ms = 0.0;
    const double mtp0 = mtp_ms_;
    bool go = true;
    // STRATA_DSV4_MTP_GATE=<p> (default 0.5, 0: off): a draft whose softmax probability is below p is not verified - a
    // plain step instead (measured: rounds under 0.5 were accepted 25-45%, at 0.95+ 97%; two cooled rounds 34.9 / 34.8
    // -> 35.4 / 35.2 tok/s) (a
    // rejected round costs a two-row pass for one token); STRATA_DSV4_MTP_CONFLOG=<file>: "conf accepted" a round
    static const float gate = getenv("STRATA_DSV4_MTP_GATE") ? (float) atof(getenv("STRATA_DSV4_MTP_GATE")) : 0.5f;
    static FILE* conflog = getenv("STRATA_DSV4_MTP_CONFLOG") ? std::fopen(getenv("STRATA_DSV4_MTP_CONFLOG"), "a") : nullptr;
    std::vector<float> lg;
    while (go) {
        int ya = -1, yb = -1;
        const int64_t p = pos_;
        if (gate > 0.0f && d4_mtp_conf_ < gate) {   // the plain step: a's row, then the next draft from its streams
            ++d4_gated_;
            if (!forward({(int32_t) a}, lg, err)) return false;
            F->tokens += 1;
            int y = sample_token(sp, err);
            if (y < 0) return false;
            y = forced(y);
            sp.counter += 1;
            ++produced;
            last_tok_ = y;
            if (!emit(y)) break;
            a = y;
            if (!d4_mtp_draft(a, pos_, dft, err)) return false;
            continue;
        }
        const float conf = d4_mtp_conf_;
        const auto tv = std::chrono::steady_clock::now();
        if (!d4_verify2(a, dft, ya, yb, err)) return false;
        if (conflog) std::fprintf(conflog, "%.4f %d\n", conf, forced(ya) == dft && ya == dft ? 1 : 0);
        verify_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tv).count();
        ++d4_rounds_;
        const int t1 = forced(ya);
        if (t1 == dft && ya == dft) {   // the draft holds: both rows stay
            ++d4_accepts_;
            sp.counter += 1;
            ++produced;
            if (!emit(t1)) { rollback(); pos_ = p + 1; last_tok_ = ya; break; }
            pos_ = p + 2;
            const int t2 = forced(yb);
            sp.counter += 1;
            ++produced;
            F->tokens += 2;
            int dummy = -1;
            // the block's window takes row (d at p + 1, A's streams) before it drafts from (t2, B's)
            if (!d4_mtp_draft(dft, p + 1, dummy, err, true)) return false;
            cudaMemcpyAsync(state_, B->Rb, hcb, cudaMemcpyDeviceToDevice, s);
            last_tok_ = t2;
            if (!emit(t2)) break;
            a = t2;
            if (!d4_mtp_draft(a, p + 2, dft, err)) return false;
        } else {
            rollback();
            pos_ = p + 1;
            F->tokens += 1;
            sp.counter += 1;
            ++produced;
            last_tok_ = t1;
            if (!emit(t1)) break;
            a = t1;
            if (!d4_mtp_draft(a, p + 1, dft, err)) return false;
        }
        (void) max_new;
    }
    cudaStreamSynchronize(s);
    F->ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    (void) tok0;
    const double wall = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const double nr = (double) std::max<uint64_t>(1, d4_rounds_ - r0);
    if (conflog) std::fflush(conflog);
    std::fprintf(stderr, "dsv4 mtp spec: %llu plain steps by the gate\n", (unsigned long long) d4_gated_);
    std::fprintf(stderr, "dsv4 mtp spec: %llu / %llu drafts accepted (%.1f%%); a round %.1f ms: verify %.1f, drafts %.1f\n",
                 (unsigned long long) d4_accepts_, (unsigned long long) d4_rounds_,
                 100.0 * d4_accepts_ / std::max<uint64_t>(1, d4_rounds_), wall / nr, verify_ms / nr, (mtp_ms_ - mtp0) / nr);
    return true;
}
}  // namespace strata::core
