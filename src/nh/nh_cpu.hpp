// src/nh/nh_cpu.hpp - maya-nh's CPU lane: a token's routed experts that are not in VRAM, computed on the CPU from host
// memory while the GPU computes the others (the shared CPU EXL3 kernels, include/strata/kernels/cpu/exl3_cpu.hpp, with
// Nemotron's relu^2 between up and down - no gate).
#pragma once

#include <functional>
#include <vector>

#include "strata/kernels/cpu/exl3_cpu.hpp"

namespace nh {

struct CpuExpert {
    strata::kernels::cpu::Exl3Part up, down;
    float w = 0.0f;   // its weight in the sum (routed_scaling_factor included)
};

class CpuLane {
public:
    /// threads: workers besides the caller; lat: the experts' input / output width, ff: their hidden width (multiples
    /// of 128).
    CpuLane(int threads, int lat, int ff);
    ~CpuLane();
    CpuLane(const CpuLane&) = delete;
    CpuLane& operator=(const CpuLane&) = delete;
    /// out [lat] = sum over ex (in order) of w * down(relu(up(x))^2), x [lat] F32.  At most 32 experts.
    void run(const std::vector<CpuExpert>& ex, const float* x, float* out);
    int threads() const;
    /// The workers a lane takes here: one a physical core but the caller's (SMT siblings left out), 2 cores left free.
    static int auto_workers();

private:
    struct Impl;
    Impl* p;
};

}  // namespace nh
