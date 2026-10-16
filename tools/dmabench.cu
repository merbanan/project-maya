// tools/dmabench.cu - build: nvcc -arch=sm_70 -O2 -std=c++17 tools/dmabench.cu -o dmabench.  V100 PCIe3 (2026-10-16): copy engine
// 13.0 GB/s, gather kernel 13.2 (80 blocks) / 13.1 (16 blocks); both overlap a compute kernel on another stream fully.
// copy-engine (cudaMemcpyAsync) vs SM kernel reading mapped pinned memory: N experts of B bytes host -> device,
// alone, and each concurrent with a compute kernel on another stream (the overlap question)
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <algorithm>
#include <vector>
__global__ void gather(const uint4* __restrict__ src, uint4* __restrict__ dst, size_t n) {
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) dst[i] = src[i];
}
__global__ void burn(float* x, int iters) {   // a memory + ALU-bound stand-in for a decode layer's GEMV work
    float v = x[blockIdx.x * blockDim.x + threadIdx.x];
    for (int i = 0; i < iters; ++i) v = v * 1.000001f + 0.5f;
    x[blockIdx.x * blockDim.x + threadIdx.x] = v;
}
int main() {
    const int N = 22; const size_t B = 4u << 20;   // 22 experts of 4 MiB (GLM: ~22 RAM fetches a token)
    uint8_t* h; cudaHostAlloc(&h, N * B, cudaHostAllocMapped); for (size_t i = 0; i < N * B; ++i) h[i] = (uint8_t) i;
    uint8_t* hd; cudaHostGetDevicePointer((void**) &hd, h, 0);
    uint8_t* d; cudaMalloc(&d, N * B); float* w; cudaMalloc(&w, 80 * 1024 * 4); cudaMemset(w, 0, 80 * 1024 * 4);
    cudaStream_t s1, s2; cudaStreamCreate(&s1); cudaStreamCreate(&s2);
    cudaEvent_t a, b, c; cudaEventCreate(&a); cudaEventCreate(&b); cudaEventCreate(&c);
    auto copy_ce = [&](cudaStream_t s) { for (int e = 0; e < N; ++e) cudaMemcpyAsync(d + e * B, h + e * B, B, cudaMemcpyHostToDevice, s); };
    auto copy_k = [&](cudaStream_t s, int blocks) { gather<<<blocks, 256, 0, s>>>((const uint4*) hd, (uint4*) d, N * B / 16); };
    for (int warm = 0; warm < 3; ++warm) { copy_ce(s1); copy_k(s1, 80); }
    cudaDeviceSynchronize();
    float ms;
    auto timeit = [&](const char* name, auto fn) {
        float best = 1e9;
        for (int r = 0; r < 5; ++r) { cudaEventRecord(a, s1); fn(); cudaEventRecord(b, s1); cudaEventSynchronize(b); cudaEventElapsedTime(&ms, a, b); best = std::min(best, ms); }
        std::printf("%-44s %7.2f ms  (%.1f GB/s)\n", name, best, N * B / best / 1e6);
    };
    timeit("copy engine, alone", [&] { copy_ce(s1); });
    timeit("SM gather kernel (80 blocks), alone", [&] { copy_k(s1, 80); });
    timeit("SM gather kernel (16 blocks), alone", [&] { copy_k(s1, 16); });
    // compute alone, then compute + each copy form on another stream
    auto burn_only = [&] { burn<<<80, 1024, 0, s1>>>(w, 200000); };
    timeit("compute kernel alone", burn_only);
    timeit("compute + copy engine (other stream)", [&] { cudaEventRecord(c, s1); cudaStreamWaitEvent(s2, c, 0); copy_ce(s2); burn_only(); cudaEventRecord(c, s2); cudaStreamWaitEvent(s1, c, 0); });
    timeit("compute + SM gather 16 blocks (other stream)", [&] { cudaEventRecord(c, s1); cudaStreamWaitEvent(s2, c, 0); copy_k(s2, 16); burn_only(); cudaEventRecord(c, s2); cudaStreamWaitEvent(s1, c, 0); });
    return 0;
}
