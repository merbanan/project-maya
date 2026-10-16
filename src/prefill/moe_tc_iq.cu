// src/prefill/moe_tc_iq.cu - see include/strata/prefill/moe_tc_iq.hpp: the prompt's routed experts on Volta's FP16
// tensor cores for the IQ formats of the GLM / Hy3 packs (IQ2_XXS, IQ2_S, IQ3_XXS).
//
// jmnargi/Strata-V100's moe_fp16tc shape (31ad437c): a block computes 64 routed rows x 128 output columns, eight warps of
// 32 x 32, the reduction in steps of 64 values whose weights (128 rows x 64) and activations (64 rows x 64, the q8_1 rows
// MMQ quantizes - D4 for these types) are decoded to FP16 in shared memory, WMMA m16n16k16, FP32 accumulation.  A thread
// decodes 32 values (one ib32 of a 256-value super-block) with ggml's own grids (dequantize_row_iq*).  Not bit-identical
// to MMQ (FP16 operands).
#include "strata/prefill/moe_tc_iq.hpp"

#include "common.cuh"   // ggml-cuda: ggml-common.h's grids as __device__ tables

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strata::prefill::tciq {
namespace {

constexpr int BM = 64, BN = 128, KS = 64, TS = 256;
constexpr int AS = 80, WS = 72;
constexpr int ACTB = 144;
constexpr int T_IQ2_XXS = 16, T_IQ3_XXS = 18, T_IQ2_S = 22;

template <int TYPE> __host__ __device__ constexpr int sb_bytes() { return TYPE == T_IQ2_XXS ? 66 : TYPE == T_IQ3_XXS ? 98 : 82; }

// 32 values (ib32 of the super-block B) as FP16 into o - ggml's dequantize_row_iq2_xxs / iq3_xxs / iq2_s
template <int TYPE>
__device__ __forceinline__ void dequant32(const uint8_t* B, int ib32, __half* o) {
    const float d = __half2float(*(const __half*) B);
    if constexpr (TYPE == T_IQ2_XXS) {
        const uint16_t* qs = (const uint16_t*) (B + 2) + 4 * ib32;
        const uint32_t a0 = (uint32_t) qs[0] | ((uint32_t) qs[1] << 16), a1 = (uint32_t) qs[2] | ((uint32_t) qs[3] << 16);
        const float db = d * (0.5f + (float) (a1 >> 28)) * 0.25f;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint8_t* grid = (const uint8_t*) (iq2xxs_grid + ((a0 >> (8 * l)) & 0xFF));
            const uint8_t signs = ksigns_iq2xs[(a1 >> (7 * l)) & 127];
#pragma unroll
            for (int j = 0; j < 8; ++j)
                o[8 * l + j] = __float2half(db * (float) grid[j] * ((signs & kmask_iq2xs[j]) ? -1.0f : 1.0f));
        }
    } else if constexpr (TYPE == T_IQ3_XXS) {
        const uint8_t* qs = B + 2 + 8 * ib32;
        const uint8_t* sas = B + 2 + 64;
        const uint32_t a = (uint32_t) sas[4 * ib32] | ((uint32_t) sas[4 * ib32 + 1] << 8) |
                           ((uint32_t) sas[4 * ib32 + 2] << 16) | ((uint32_t) sas[4 * ib32 + 3] << 24);
        const float db = d * (0.5f + (float) (a >> 28)) * 0.5f;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint8_t signs = ksigns_iq2xs[(a >> (7 * l)) & 127];
            const uint8_t* g1 = (const uint8_t*) (iq3xxs_grid + qs[2 * l]);
            const uint8_t* g2 = (const uint8_t*) (iq3xxs_grid + qs[2 * l + 1]);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                o[8 * l + j] = __float2half(db * (float) g1[j] * ((signs & kmask_iq2xs[j]) ? -1.0f : 1.0f));
                o[8 * l + j + 4] = __float2half(db * (float) g2[j] * ((signs & kmask_iq2xs[j + 4]) ? -1.0f : 1.0f));
            }
        }
    } else {   // IQ2_S: d, qs[64] (32 grid bytes, 32 sign bytes), qh[8], scales[8]
        const uint8_t* qs = B + 2 + 4 * ib32;
        const uint8_t* signs = B + 2 + 32 + 4 * ib32;
        const uint8_t qh = B[2 + 64 + ib32];
        const uint8_t sc = B[2 + 64 + 8 + ib32];
        const float db0 = d * (0.5f + (float) (sc & 0xF)) * 0.25f, db1 = d * (0.5f + (float) (sc >> 4)) * 0.25f;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const float dl = l < 2 ? db0 : db1;
            const uint8_t* grid = (const uint8_t*) (iq2s_grid + (qs[l] | ((qh << (8 - 2 * l)) & 0x300)));
#pragma unroll
            for (int j = 0; j < 8; ++j)
                o[8 * l + j] = __float2half(dl * (float) grid[j] * ((signs[l] & kmask_iq2xs[j]) ? -1.0f : 1.0f));
        }
    }
}

template <int TYPE>
__global__ void __launch_bounds__(TS, 1) tc_iq_kernel(const uint8_t* __restrict__ W, size_t expert_bytes, size_t row_bytes,
                                                     int n_in, const int32_t* __restrict__ bounds,
                                                     const uint8_t* __restrict__ xq, int64_t xq_rows,
                                                     float* __restrict__ dst, int64_t ld) {
    using namespace nvcuda;
    const int z = blockIdx.z;
    const int lo = bounds[z], hi = bounds[z + 1];
    const int row0 = lo + (int) blockIdx.y * BM;
    if (row0 >= hi) return;
    const int local = min(BM, hi - row0);
    const int col0 = (int) blockIdx.x * BN;
    const int tid = threadIdx.x, warp = tid >> 5;
    const int m_base = (warp >> 2) * 32, n_base = (warp & 3) * 32;
    __shared__ __align__(32) union {
        struct { __half a[BM * AS]; __half w[BN * WS]; } ab;
        float c[BM * BN];
    } sm;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
        for (int c = 0; c < 2; ++c) wmma::fill_fragment(acc[r][c], 0.0f);
    const uint8_t* We = W + (size_t) z * expert_bytes;
    for (int k0 = 0; k0 < n_in; k0 += KS) {
        __syncthreads();
        {
            const int r = tid >> 1, h = tid & 1;
            const uint8_t* B = We + (size_t) (col0 + r) * row_bytes + (size_t) (k0 / 256) * sb_bytes<TYPE>();
            dequant32<TYPE>(B, (k0 % 256) / 32 + h, sm.ab.w + (size_t) r * WS + 32 * h);
        }
        {
            const int m = tid >> 2, part = tid & 3;
            __half* o = sm.ab.a + (size_t) m * AS + part * 16;
            if (m >= local) {
#pragma unroll
                for (int j = 0; j < 16; ++j) o[j] = __float2half(0.0f);
            } else {
                const int64_t row = (int64_t) row0 + m;
                const int off = (k0 % 128) + part * 16;
                const uint8_t* base = xq + ((size_t) (k0 / 128) * (size_t) xq_rows + (size_t) row) * ACTB;
                const float dsc = ((const float*) base)[off >> 5];   // D4
                const int8_t* qv = (const int8_t*) (base + 16 + off);
#pragma unroll
                for (int j = 0; j < 16; ++j) o[j] = __float2half(dsc * (float) qv[j]);
            }
        }
        __syncthreads();
#pragma unroll
        for (int k16 = 0; k16 < KS / 16; ++k16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> bf[2];
#pragma unroll
            for (int r = 0; r < 2; ++r) wmma::load_matrix_sync(af[r], &sm.ab.a[(size_t) (m_base + 16 * r) * AS + k16 * 16], AS);
#pragma unroll
            for (int c = 0; c < 2; ++c) wmma::load_matrix_sync(bf[c], &sm.ab.w[(size_t) (n_base + 16 * c) * WS + k16 * 16], WS);
#pragma unroll
            for (int r = 0; r < 2; ++r)
#pragma unroll
                for (int c = 0; c < 2; ++c) wmma::mma_sync(acc[r][c], af[r], bf[c], acc[r][c]);
        }
    }
    __syncthreads();
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
        for (int c = 0; c < 2; ++c)
            wmma::store_matrix_sync(&sm.c[(size_t) (m_base + 16 * r) * BN + n_base + 16 * c], acc[r][c], BN,
                                    wmma::mem_row_major);
    __syncthreads();
    for (int i = tid; i < BM * BN; i += TS) {
        const int m = i / BN;
        if (m < local) dst[(size_t) (row0 + m) * ld + col0 + (i % BN)] = sm.c[i];
    }
}

// ---- the register-fragment kernel (the default; STRATA_GLM_PREFILL_TC_RF=0: the shared-memory WMMA one above).
// The design of the Strata V100 tree's moe_fp16tc expert_kernel_rf: mma.m8n8k4 splits a warp into four quad pairs, each
// an 8 x 8 x 4 product with a lane holding one row of A, one column of B and 8 FP32 sums.  A block is 64 routed rows by
// two weight sets of 64 rows (pair: rows r and n_out / 2 + r - gate and up; else rows r and r + 64 of a 128-row block),
// eight warps of 8 rows a set.  A k step is 128 values (one q8_1 block); quad pair q takes the step's ib32 q - its lane
// decodes the row's 32 values (dequant32, ggml's grids) straight into eight A fragments, no shared-memory weights.  The
// activations are staged once a step as FP16 (double-buffered), the four quarters summed by shuffles at the end.
constexpr int RT = 64, RK = 128, RAS = RK + 8;

__device__ __forceinline__ void mma884(float* c, uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
                 "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t h2u(__half2 h) { return *reinterpret_cast<const uint32_t*>(&h); }

// one thread's share of a step's activations: row tid / 4 of the tile, values [32 (tid % 4), +32) of the q8_1 block
struct RawA { uint4 q0, q1; float d; };
__device__ __forceinline__ RawA load_a(const uint8_t* xq, int64_t xq_rows, int64_t row, bool live, int kb, int part) {
    RawA r;
    if (!live) { r.q0 = make_uint4(0, 0, 0, 0); r.q1 = r.q0; r.d = 0.0f; return r; }
    const uint8_t* base = xq + ((size_t) kb * (size_t) xq_rows + (size_t) row) * ACTB;
    r.d = __ldg((const float*) base + part);
    const uint4* q = reinterpret_cast<const uint4*>(base + 16 + 32 * part);
    r.q0 = __ldg(q);
    r.q1 = __ldg(q + 1);
    return r;
}
__device__ __forceinline__ void store_a(const RawA& r, __half* o) {
    const __half2 d2 = __float2half2_rn(r.d);
    const uint32_t c1152u = 0x64806480u;
    const __half2 c1152 = *reinterpret_cast<const __half2*>(&c1152u);
    const uint32_t w[8] = {r.q0.x, r.q0.y, r.q0.z, r.q0.w, r.q1.x, r.q1.y, r.q1.z, r.q1.w};
    uint32_t h[16];
#pragma unroll
    for (int k = 0; k < 8; ++k) {   // the exact signed code from a mantissa bias: {0x64, byte ^ 0x80} = 1152 + code
        const uint32_t v = w[k] ^ 0x80808080u;
        const uint32_t lo = __byte_perm(v, 0x64646464u, 0x4140), hi = __byte_perm(v, 0x64646464u, 0x4342);
        h[2 * k] = h2u(__hmul2(__hsub2(*reinterpret_cast<const __half2*>(&lo), c1152), d2));
        h[2 * k + 1] = h2u(__hmul2(__hsub2(*reinterpret_cast<const __half2*>(&hi), c1152), d2));
    }
    uint4* o4 = reinterpret_cast<uint4*>(o);
#pragma unroll
    for (int i = 0; i < 4; ++i) o4[i] = make_uint4(h[4 * i], h[4 * i + 1], h[4 * i + 2], h[4 * i + 3]);
}

template <int TYPE>
__device__ __forceinline__ void unpack_w(const uint8_t* row, int k, uint32_t* a) {
    __align__(16) __half v[32];
    dequant32<TYPE>(row + (size_t) (k >> 8) * sb_bytes<TYPE>(), (k & 255) >> 5, v);
#pragma unroll
    for (int t = 0; t < 16; ++t) a[t] = h2u(__halves2half2(v[2 * t], v[2 * t + 1]));
}

template <int TYPE>
__global__ void __launch_bounds__(TS, 1) tc_iq_rf_kernel(const uint8_t* __restrict__ W, size_t expert_bytes, size_t row_bytes,
                                                        int n_in, int n_out, bool pair, const int32_t* __restrict__ bounds,
                                                        const uint8_t* __restrict__ xq, int64_t xq_rows,
                                                        float* __restrict__ dst, int64_t ld) {
    const int z = blockIdx.z;
    const int lo = bounds[z], hi = bounds[z + 1];
    const int row0 = lo + (int) blockIdx.y * RT;
    if (row0 >= hi) return;
    const int local = min(RT, hi - row0), ngrp = (local + 7) >> 3;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int qp = (lane >> 2) & 3, j = (lane & 3) | ((lane >> 4) << 2);
    const int cb0 = (int) blockIdx.x * (pair ? 64 : 128) + 8 * warp;   // the warp's first row of set 0
    const int cb1 = pair ? n_out / 2 + cb0 : cb0 + 64;
    const uint8_t* We = W + (size_t) z * expert_bytes;
    const uint8_t* w0 = We + (size_t) (cb0 + j) * row_bytes;
    const uint8_t* w1 = We + (size_t) (cb1 + j) * row_bytes;
    __shared__ __align__(16) __half sa[2][RT * RAS];
    float c[2][8][8];
#pragma unroll
    for (int s2 = 0; s2 < 2; ++s2)
#pragma unroll
        for (int gi = 0; gi < 8; ++gi)
#pragma unroll
            for (int i = 0; i < 8; ++i) c[s2][gi][i] = 0.0f;
    const int am = tid >> 2, apart = tid & 3;
    const bool alive = am < local;
    const int64_t arow = (int64_t) row0 + am;
    const int nsteps = n_in / RK;
    RawA ra = load_a(xq, xq_rows, arow, alive, 0, apart);
    store_a(ra, &sa[0][am * RAS + 32 * apart]);
    __syncthreads();
    for (int st = 0; st < nsteps; ++st) {
        const int buf = st & 1;
        if (st + 1 < nsteps) ra = load_a(xq, xq_rows, arow, alive, st + 1, apart);   // (in flight under the products)
        uint32_t a0[16], a1[16];
        unpack_w<TYPE>(w0, st * RK + 32 * qp, a0);
        unpack_w<TYPE>(w1, st * RK + 32 * qp, a1);
        const __half* sb = sa[buf];
#pragma unroll
        for (int gi = 0; gi < 8; ++gi) {
            if (gi < ngrp) {
                const uint4* bp = reinterpret_cast<const uint4*>(sb + (8 * gi + j) * RAS + 32 * qp);
                uint4 bv[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) bv[i] = bp[i];
                const uint32_t* bw = reinterpret_cast<const uint32_t*>(bv);
#pragma unroll
                for (int kk = 0; kk < 8; ++kk) {
                    mma884(c[0][gi], a0[2 * kk], a0[2 * kk + 1], bw[2 * kk], bw[2 * kk + 1]);
                    mma884(c[1][gi], a1[2 * kk], a1[2 * kk + 1], bw[2 * kk], bw[2 * kk + 1]);
                }
            }
        }
        if (st + 1 < nsteps) store_a(ra, &sa[buf ^ 1][am * RAS + 32 * apart]);
        __syncthreads();
    }
#pragma unroll
    for (int s2 = 0; s2 < 2; ++s2)
#pragma unroll
        for (int gi = 0; gi < 8; ++gi)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                float v = c[s2][gi][i];
                v += __shfl_xor_sync(0xffffffffu, v, 4);
                v += __shfl_xor_sync(0xffffffffu, v, 8);
                c[s2][gi][i] = v;
            }
    if (qp != 0) return;
#pragma unroll
    for (int gi = 0; gi < 8; ++gi) {
        if (gi >= ngrp) continue;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int rr = (lane & 1) + (i & 2) + ((lane >> 4) << 2);   // the weight row in the 8
            const int tt = 8 * gi + (i & 4) + (lane & 2) + (i & 1);      // the routed row in the tile
            if (tt < local) {
                float* o = dst + (size_t) (row0 + tt) * ld;
                o[cb0 + rr] = c[0][gi][i];
                o[cb1 + rr] = c[1][gi][i];
            }
        }
    }
}

}  // namespace

bool supported(int type, int64_t n_in, int64_t n_out) {
    return (type == T_IQ2_XXS || type == T_IQ3_XXS || type == T_IQ2_S) && n_in % 256 == 0 && n_out % BN == 0;
}

bool available() {
    static const int ok = [] {
        int dev = 0, major = 0, minor = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) return 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
        return major == 7 && minor == 0 ? 1 : 0;   // (the MMQ int8 tensor-core kernels serve sm_75+)
    }();
    return ok != 0;
}

void product(int type, const void* W, size_t expert_bytes, int64_t n_in, int64_t n_out, int n_experts, const int32_t* bounds,
             int max_rows, const void* xq, int64_t xq_rows, float* dst, int64_t ld, void* stream, bool pair) {
    if (!supported(type, n_in, n_out)) throw std::invalid_argument("moe_tc_iq: type / shape");
    if (max_rows <= 0 || n_experts <= 0) return;
    const int sbb = type == T_IQ2_XXS ? 66 : type == T_IQ3_XXS ? 98 : 82;
    const size_t rb = (size_t) (n_in / 256) * sbb;
    cudaStream_t s = (cudaStream_t) stream;
    const uint8_t* w = (const uint8_t*) W;
    const uint8_t* x = (const uint8_t*) xq;
    static const bool rf = !(getenv("STRATA_GLM_PREFILL_TC_RF") && atoi(getenv("STRATA_GLM_PREFILL_TC_RF")) == 0);
    if (rf && n_in % RK == 0 && (pair ? n_out % 128 == 0 : n_out % 128 == 0)) {
        const dim3 g2((unsigned) (pair ? n_out / 2 / 64 : n_out / 128), (unsigned) ((max_rows + RT - 1) / RT), (unsigned) n_experts);
        const int ni = (int) n_in, no = (int) n_out;
        if (type == T_IQ2_XXS) tc_iq_rf_kernel<T_IQ2_XXS><<<g2, TS, 0, s>>>(w, expert_bytes, rb, ni, no, pair, bounds, x, xq_rows, dst, ld);
        else if (type == T_IQ3_XXS) tc_iq_rf_kernel<T_IQ3_XXS><<<g2, TS, 0, s>>>(w, expert_bytes, rb, ni, no, pair, bounds, x, xq_rows, dst, ld);
        else tc_iq_rf_kernel<T_IQ2_S><<<g2, TS, 0, s>>>(w, expert_bytes, rb, ni, no, pair, bounds, x, xq_rows, dst, ld);
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) throw std::runtime_error(std::string("moe_tc_iq rf: ") + cudaGetErrorString(e));
        return;
    }
    const dim3 grid((unsigned) (n_out / BN), (unsigned) ((max_rows + BM - 1) / BM), (unsigned) n_experts);
    if (type == T_IQ2_XXS) tc_iq_kernel<T_IQ2_XXS><<<grid, TS, 0, s>>>(w, expert_bytes, rb, (int) n_in, bounds, x, xq_rows, dst, ld);
    else if (type == T_IQ3_XXS) tc_iq_kernel<T_IQ3_XXS><<<grid, TS, 0, s>>>(w, expert_bytes, rb, (int) n_in, bounds, x, xq_rows, dst, ld);
    else tc_iq_kernel<T_IQ2_S><<<grid, TS, 0, s>>>(w, expert_bytes, rb, (int) n_in, bounds, x, xq_rows, dst, ld);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("moe_tc_iq: ") + cudaGetErrorString(e));
}

}  // namespace strata::prefill::tciq
