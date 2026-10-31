// q35_attn micro benchmark: a random K/V cache of N positions (FP16 and K8V4), T query tokens at the end,
// the time a call and the K/V bytes a second; and a check of the K8V4 output against the FP16 one.
// build: nvcc -arch=sm_70 -O3 -I include tests/q35/attn_bench.cu src/kernels/cuda/q35_kernels.cu src/kernels/cuda/dense_kernels.cu
#include "strata/kernels/q35_kernels.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace strata::kernels;

// the GPU's streaming read rate over a buffer: every 16 bytes read once (uint4 loads, grid-stride), XOR-folded so the
// reads are not dropped - the ceiling the attention's K / V reads are measured against
__global__ void read_kernel(const uint4* __restrict__ a, size_t n, unsigned* out) {
    unsigned x = 0;
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        const uint4 v = __ldg(a + i);
        x ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (x == 0x12345678u) *out = x;
}
double read_gbs(const void* buf, size_t bytes, cudaStream_t s) {
    unsigned* o;
    cudaMalloc(&o, 4);
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    double best = 0;
    for (int blocks : {sms * 4, sms * 8, sms * 16, sms * 32}) {
        read_kernel<<<blocks, 256, 0, s>>>((const uint4*) buf, bytes / 16, o);
        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
        cudaEventRecord(a, s);
        for (int r = 0; r < 10; ++r) read_kernel<<<blocks, 256, 0, s>>>((const uint4*) buf, bytes / 16, o);
        cudaEventRecord(b, s); cudaEventSynchronize(b);
        float ms = 0; cudaEventElapsedTime(&ms, a, b);
        best = std::max(best, bytes / (ms / 10 * 1e-3) / 1e9);
    }
    cudaFree(o);
    return best;
}

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
    {
        const size_t kb = (size_t) HK * max_ctx * D * 2;
        printf("streaming read (the ceiling): K cache %.0f GB/s, V cache %.0f GB/s (%.0f MB each)\n", read_gbs(kv[0].k, kb, s),
               read_gbs(kv[0].v, kb, s), kb / 1e6);
    }
    // the streaming kernel (FP16): blocks, against the tile kernel's output
    for (int T : {1, 2, 4, 5}) {
        const int p0 = N - T;
        q35_set_attn_stream(0, 0, 0);
        q35_set_attn_dec_max_tokens(0);
        q35_set_attn_var(0);
        q35_attn_prep(qf, H * 2 * D, kk, vv, HK * D, nw, nw, 1e-6f, 1e7f, 64, p0, T, H, HK, D, qo, kv[0], s);
        q35_attn(qo, kv[0], p0, T, H, HK, D, qf, H * 2 * D, out, scr, 0, s);
        std::vector<float> ref((size_t) T * H * D), o2(ref.size());
        cudaMemcpy(ref.data(), out, ref.size() * 4, cudaMemcpyDeviceToHost);
        for (int cfg = 0; cfg < 6; ++cfg) {
            const int sk = cfg < 3 ? 64 : 32, bl = (cfg % 3 == 0) ? 0 : (cfg % 3 == 1 ? 160 : 320);
            q35_set_attn_stream(1, bl, sk);
            cudaMemset(out, 0, ref.size() * 4);
            q35_attn(qo, kv[0], p0, T, H, HK, D, qf, H * 2 * D, out, scr, 0, s);
            cudaMemcpy(o2.data(), out, o2.size() * 4, cudaMemcpyDeviceToHost);
            double e = 0, r = 0;
            for (size_t i = 0; i < ref.size(); ++i) { e += pow(o2[i] - ref[i], 2); r += ref[i] * ref[i]; }
            cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
            cudaEventRecord(a, s);
            for (int rep = 0; rep < 20; ++rep) q35_attn(qo, kv[0], p0, T, H, HK, D, qf, H * 2 * D, out, scr, 0, s);
            cudaEventRecord(b, s); cudaEventSynchronize(b);
            float ms = 0; cudaEventElapsedTime(&ms, a, b); ms /= 20;
            const double gb16 = (double) N * HK * D * 2 * 2 / 1e9;
            printf("N %d T %d stream sk %d blocks %3d: fp16 %.3f ms (%.0f GB/s)  vs tile rel err %.2e  %s\n", N, T, sk, bl, ms,
                   gb16 / ms * 1e3, sqrt(e / r), cudaGetErrorString(cudaGetLastError()));
        }
        q35_set_attn_stream(0, 0, 0);
    }
    // the decode kernel's tuning: keys in flight a warp x target blocks, at T 1 and 2
    for (int T : {1}) {
        q35_set_attn_dec_max_tokens(8);
        q35_set_attn_var(0);
        const int p0 = N - T;
        for (int mb : {1})
        for (int pf : {2})
            for (int bl : {320}) {
                if (mb == 4 && pf == 8) continue;
                q35_set_attn_dec_tune(pf, bl, mb);
                float ms[2] = {0, 0};
                for (int m = 0; m < 2; ++m) {
                    q35_attn(qo, kv[m], p0, T, H, HK, D, qf, H * 2 * D, out, scr, 0, s);
                    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
                    cudaEventRecord(a, s);
                    for (int r = 0; r < 20; ++r) q35_attn(qo, kv[m], p0, T, H, HK, D, qf, H * 2 * D, out, scr, 0, s);
                    cudaEventRecord(b, s); cudaEventSynchronize(b);
                    cudaEventElapsedTime(&ms[m], a, b); ms[m] /= 20;
                }
                const double gb16 = (double) N * HK * D * 2 * 2 / 1e9, gb8 = (double) N * HK * (D + D / 2 + 32) / 1e9;
                printf("N %d T %d dec minb %d pf %d blocks %4d: fp16 %.3f ms (%.0f GB/s)  k8v4 %.3f ms (%.0f GB/s)  %s\n", N, T, mb, pf, bl,
                       ms[0], gb16 / ms[0] * 1e3, ms[1], gb8 / ms[1] * 1e3, cudaGetErrorString(cudaGetLastError()));
            }
        q35_set_attn_dec_tune(0, 0, 0);
    }
    for (int T : {1, 2, 4, 8}) {
        const int p0 = N - T;
        std::vector<float> ref;
        for (int dec : {0, 8, 101, 102}) {   // 0 the tile kernel, 8 the decode kernel, 10x tile variant x
            if (dec > 100 && T == 1) continue;
            q35_set_attn_dec_max_tokens(dec > 100 ? 0 : dec);
            q35_set_attn_var(dec > 100 ? dec - 100 : 0);
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
                   N, T, dec == 8 ? "dec  " : dec > 100 ? (dec == 101 ? "var1 " : "var2 ") : "tile ", ms[0], gb16 / ms[0] * 1e3, ms[1], gb8 / ms[1] * 1e3, sqrt(e / r),
                   dec ? sqrt(e2 / (2 * r)) : 0.0, cudaGetErrorString(cudaGetLastError()));
        }
    }
    return 0;
}
