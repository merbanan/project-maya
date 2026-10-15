// q35_attn micro benchmark: a random K/V cache of N positions (FP16 and K8V4), T query tokens at the end,
// the time a call and the K/V bytes a second; and a check of the K8V4 output against the FP16 one.
// build: nvcc -arch=sm_70 -O3 -I include tests/q35/attn_bench.cu src/kernels/cuda/q35_kernels.cu src/kernels/cuda/dense_kernels.cu
#include "strata/kernels/q35_kernels.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace strata::kernels;

int main(int argc, char** argv) {
    const int N = argc > 1 ? atoi(argv[1]) : 90000;
    const int H = 24, HK = 4, D = 256, max_ctx = N + 16;
    std::mt19937 rng(3);
    std::normal_distribution<float> nd(0.f, 1.f);
    // K and V rows of the context, written through q35_attn_prep in chunks (both formats)
    Q35Kv kv[2];
    for (int m = 0; m < 2; ++m) {
        kv[m].mode = m; kv[m].max_ctx = max_ctx;
        const size_t cells = (size_t) HK * max_ctx;
        cudaMalloc(&kv[m].k, m ? cells * D : cells * D * 2);
        cudaMalloc(&kv[m].v, m ? cells * D / 2 : cells * D * 2);
        if (m) { cudaMalloc(&kv[m].ks, cells * 8 * 2); cudaMalloc(&kv[m].vs, cells * 8 * 2); }
    }
    const int CH = 1024;
    std::vector<float> h((size_t) CH * H * 2 * D);
    float *qf, *kk, *vv, *qo, *nw, *out, *scr;
    cudaMalloc(&qf, h.size() * 4); cudaMalloc(&kk, (size_t) CH * HK * D * 4); cudaMalloc(&vv, (size_t) CH * HK * D * 4);
    cudaMalloc(&qo, (size_t) CH * H * D * 4); cudaMalloc(&nw, D * 4);
    std::vector<float> ones(D, 1.0f); cudaMemcpy(nw, ones.data(), D * 4, cudaMemcpyHostToDevice);
    cudaMalloc(&out, (size_t) 2 * 8 * H * D * 4);
    cudaMalloc(&scr, q35_attn_scratch_floats(8, H, D, max_ctx) * 4);
    std::vector<float> kh((size_t) CH * HK * D);
    cudaStream_t s; cudaStreamCreate(&s);
    for (int p = 0; p < N; p += CH) {
        const int T = std::min(CH, N - p);
        for (auto& x : kh) x = nd(rng);
        cudaMemcpy(kk, kh.data(), kh.size() * 4, cudaMemcpyHostToDevice);
        for (auto& x : kh) x = nd(rng);
        cudaMemcpy(vv, kh.data(), kh.size() * 4, cudaMemcpyHostToDevice);
        for (int m = 0; m < 2; ++m)
            q35_attn_prep(qf, H * 2 * D, kk, vv, HK * D, nw, nw, 1e-6f, 1e7f, 64, p, T, H, HK, D, qo, kv[m], s);
    }
    for (auto& x : h) x = nd(rng) * 3.0f;   // the queries (sharper softmax)
    cudaMemcpy(qf, h.data(), h.size() * 4, cudaMemcpyHostToDevice);
    cudaStreamSynchronize(s);
    for (int T : {1, 2, 4, 8}) {
        const int p0 = N - T;
        std::vector<float> ref;
        for (int dec : {0, 8}) {
            q35_set_attn_dec_max_tokens(dec);
            float ms[2] = {0, 0};
            for (int m = 0; m < 2; ++m) {
                q35_attn_prep(qf, H * 2 * D, kk, vv, HK * D, nw, nw, 1e-6f, 1e7f, 64, p0, T, H, HK, D, qo, kv[m], s);
                q35_attn(qo, kv[m], p0, T, H, HK, D, qf, H * 2 * D, out + (size_t) m * 8 * H * D, scr, 0, s);
                cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
                cudaEventRecord(a, s);
                for (int r = 0; r < 20; ++r)
                    q35_attn(qo, kv[m], p0, T, H, HK, D, qf, H * 2 * D, out + (size_t) m * 8 * H * D, scr, 0, s);
                cudaEventRecord(b, s); cudaEventSynchronize(b);
                cudaEventElapsedTime(&ms[m], a, b); ms[m] /= 20;
            }
            std::vector<float> o((size_t) 2 * 8 * H * D);
            cudaMemcpy(o.data(), out, o.size() * 4, cudaMemcpyDeviceToHost);
            double e = 0, r = 0, e2 = 0;
            for (size_t i = 0; i < (size_t) T * H * D; ++i) { e += pow(o[i] - o[8 * H * D + i], 2); r += o[i] * o[i]; }
            if (dec == 0) ref = o;
            else for (size_t i = 0; i < (size_t) T * H * D; ++i) e2 += pow(o[i] - ref[i], 2) + pow(o[8 * H * D + i] - ref[8 * H * D + i], 2);
            const double gb16 = (double) N * HK * D * 2 * 2 / 1e9, gb8 = (double) N * HK * (D + D / 2 + 32) / 1e9;
            printf("N %d T %d %s: fp16 %.3f ms (%.0f GB/s)  k8v4 %.3f ms (%.0f GB/s)  k8v4 rel err %.4f  dec vs tile %.2e  cuda %s\n",
                   N, T, dec ? "dec " : "tile", ms[0], gb16 / ms[0] * 1e3, ms[1], gb8 / ms[1] * 1e3, sqrt(e / r),
                   dec ? sqrt(e2 / (2 * r)) : 0.0, cudaGetErrorString(cudaGetLastError()));
        }
    }
    return 0;
}
