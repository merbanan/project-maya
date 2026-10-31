// src/nh/nh_cpu.cpp - maya-nh's CPU lane (nh_cpu.hpp).
//
// A call runs in four phases, each split over the lane's threads (the caller one of them):
//   1. each expert's up input: xh = H(x * suh_up)                         (one job an expert)
//   2. up's bands: u[128 b ..] = up(x)                                     (one job an expert's band of 128 outputs)
//   3. relu^2, then the down input: hh = H(relu(u)^2 * suh_down)           (one job an expert)
//   4. down's bands: y[128 b ..] = down(h)                                 (one job an expert's band)
// then out = sum of w * y in the experts' order, on the caller - the same bits whatever the threads.
#include "nh_cpu.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>

#include "mm3_lane_plan.hpp"
#include "strata/kernels/cpu/pool.hpp"

#if defined(__linux__)
#include <sched.h>
#endif

namespace nh {
namespace kc = strata::kernels::cpu;

struct CpuLane::Impl {
    static constexpr int kMaxE = 32;
    int lat = 0, ff = 0;
    std::vector<std::thread> th;
    std::mutex mu;
    std::condition_variable cv, cv_done;
    // one call's jobs, on the caller's stack: published (cur) under the lock and withdrawn before the call returns,
    // so a worker only ever runs the jobs of the call it joined
    struct Call {
        const std::function<void(int)>* f = nullptr;
        int n = 0;
        std::atomic<int> next{0}, done{0};
        int active = 0;   // (workers inside it; under mu)
    };
    Call* cur = nullptr;
    uint64_t gen = 0;   // (under mu)
    bool stop = false;
    // per expert: xh (lat), xsum, u (ff), hh (ff), hsum, y (lat)
    std::vector<float> xh, u, hh, y;
    float xsum[kMaxE] = {}, hsum[kMaxE] = {};

    void worker() {
        uint64_t seen = 0;
        for (;;) {
            Call* c = nullptr;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return stop || gen != seen; });
                if (stop) return;
                seen = gen;
                c = cur;
                if (c == nullptr) continue;   // (the call it was woken for has ended)
                ++c->active;
            }
            for (int j; (j = c->next.fetch_add(1)) < c->n;) {
                (*c->f)(j);
                c->done.fetch_add(1);
            }
            {
                std::lock_guard<std::mutex> lk(mu);
                --c->active;
            }
            cv_done.notify_all();
        }
    }
    void parallel(int n, const std::function<void(int)>& f) {
        if (n <= 0) return;
        if (th.empty()) {
            for (int j = 0; j < n; ++j) f(j);
            return;
        }
        Call c;
        c.f = &f;
        c.n = n;
        {
            std::lock_guard<std::mutex> lk(mu);
            cur = &c;
            ++gen;
        }
        cv.notify_all();
        for (int j; (j = c.next.fetch_add(1)) < n;) {
            f(j);
            c.done.fetch_add(1);
        }
        std::unique_lock<std::mutex> lk(mu);
        cv_done.wait(lk, [&] { return c.done.load() >= n && c.active == 0; });
        cur = nullptr;   // (a worker waking after this finds no call)
    }
};

CpuLane::CpuLane(int threads, int lat, int ff) : p(new Impl) {
    p->lat = lat;
    p->ff = ff;
    p->xh.resize((size_t) Impl::kMaxE * lat);
    p->u.resize((size_t) Impl::kMaxE * ff);
    p->hh.resize((size_t) Impl::kMaxE * ff);
    p->y.resize((size_t) Impl::kMaxE * lat);
    for (int i = 0; i < threads; ++i) p->th.emplace_back([this] { p->worker(); });
}

CpuLane::~CpuLane() {
    {
        std::lock_guard<std::mutex> lk(p->mu);
        p->stop = true;
    }
    p->cv.notify_all();
    for (std::thread& t : p->th) t.join();
    delete p;
}

int CpuLane::threads() const { return (int) p->th.size(); }

int CpuLane::auto_workers() {
    const int w = (int) kc::physical_cores(true).size();
    int logical = 0;
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof set, &set) == 0) logical = CPU_COUNT(&set);
#endif
    if (logical <= 0) logical = (int) std::thread::hardware_concurrency();
    return mm3::lane_default_workers(w, logical, 2);
}

void CpuLane::run(const std::vector<CpuExpert>& ex, const float* x, float* out) {
    Impl& I = *p;
    const int ne = (int) std::min<size_t>(ex.size(), Impl::kMaxE);
    const int lat = I.lat, ff = I.ff, UB = ff / 128, DB = lat / 128;
    if (ne == 0) {
        std::memset(out, 0, (size_t) lat * 4);
        return;
    }
    const std::function<void(int)> f1 = [&](int e) {
        I.xsum[e] = kc::exl3_cpu_prepare(ex[(size_t) e].up, x, I.xh.data() + (size_t) e * lat);
    };
    I.parallel(ne, f1);
    const std::function<void(int)> f2 = [&](int j) {
        const int e = j / UB, b = j % UB;
        kc::exl3_cpu_band(ex[(size_t) e].up, I.xh.data() + (size_t) e * lat, I.xsum[e], b * 128,
                          I.u.data() + (size_t) e * ff + b * 128);
    };
    I.parallel(ne * UB, f2);
    const std::function<void(int)> f3 = [&](int e) {
        float* u = I.u.data() + (size_t) e * ff;
        for (int i = 0; i < ff; ++i) {
            const float r = std::max(u[i], 0.0f);
            u[i] = r * r;
        }
        I.hsum[e] = kc::exl3_cpu_prepare(ex[(size_t) e].down, u, I.hh.data() + (size_t) e * ff);
    };
    I.parallel(ne, f3);
    const std::function<void(int)> f4 = [&](int j) {
        const int e = j / DB, b = j % DB;
        kc::exl3_cpu_band(ex[(size_t) e].down, I.hh.data() + (size_t) e * ff, I.hsum[e], b * 128,
                          I.y.data() + (size_t) e * lat + b * 128);
    };
    I.parallel(ne * DB, f4);
    for (int i = 0; i < lat; ++i) {
        float a = 0.0f;
        for (int e = 0; e < ne; ++e) a += ex[(size_t) e].w * I.y[(size_t) e * lat + i];
        out[i] = a;
    }
}

}  // namespace nh
