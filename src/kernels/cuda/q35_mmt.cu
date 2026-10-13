// src/kernels/cuda/q35_mmt.cu - Y[t][n] = sum_k W[n][k] X[t][k] for up to 8 token rows on the tensor cores (sm_70+),
// the weights in their GGUF blocks (Q4_K, Q5_K, Q6_K, Q8_0).  See q35_kernels.hpp (q35_mmt).
//
// Why: the dp4a GEMV (native_mmvq) costs nearly a full weight pass per extra column - a verify window of 8 tokens
// took 2.5x a single token.  Here a weight is dequantized ONCE, to FP16 in shared memory, and one WMMA m8n32k16
// multiplies it with all 8 rows (the idea of v100-skinny's QPN kernels, github.com/dnv2003/v100-skinny): the pass
// stays a read of the weights whatever the row count up to 8.
//
// Block: 4 warps, 128 output rows (32 a warp: lane l dequantizes row n0 + l), one super-block of 256 k a step, the
// k range cut over blockIdx.y; the X slice of a step (8 x 256 FP16) is staged once a block.  Partials of the k
// slices go to `part` and q35_mmt_reduce adds them in a fixed order (deterministic).
#include "strata/kernels/q35_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

constexpr int kW = 4;                  // warps a block (128 output rows)

template <int TYPE> struct Fmt;
template <> struct Fmt<12> { static constexpr int bytes = 144; };   // Q4_K
template <> struct Fmt<13> { static constexpr int bytes = 176; };   // Q5_K
template <> struct Fmt<14> { static constexpr int bytes = 210; };   // Q6_K
template <> struct Fmt<8> { static constexpr int bytes = 272; };    // Q8_0, 8 blocks of 34 (one "super-block" of 256)


// ---- mma.sync.m8n8k4 (sm_70), .row.col, f16 x f16 -> f32.  Measured layout (probe/mma884.cu on a V100): the warp
// is 4 quadpairs (lanes 4q..4q+3 and 16+4q..16+4q+3), each an independent 8x8x4 product.  A thread's a0..a3 are row
// k 0..3 of the A row it owns, b0..b3 k 0..3 of the B column it owns; with j = lane % 4, hi = lane >= 16 and
// b = 4 * (lane / 4 % 4), its accumulators c0..c7 are (A row of lane r, B column of lane c) for
//   r in {b + (j & 1) + 16 hi, +2},  c in {b + (j & 2), +1, +16, +17}:  c0 (r0,c0) c1 (r0,c1) c2 (r1,c0) c3 (r1,c1)
//   c4 (r0,c2) c5 (r0,c3) c6 (r1,c2) c7 (r1,c3).
// Here lane L owns A row = token (L % 4) + 4 (L >= 16) and B column = output row n0 + L.
__device__ __forceinline__ void mma884(float* c, uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
                 "{%0,%1,%2,%3,%4,%5,%6,%7};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float h2f(uint32_t bits16) { return __half2float(__ushort_as_half((unsigned short) bits16)); }
__device__ __forceinline__ uint32_t h2u(__half2 h) { return *(uint32_t*) &h; }
__device__ __forceinline__ __half2 u2h(uint32_t u) { return *(__half2*) &u; }

// bytes q0..q3 (each < 256) of v -> two half2 (1024 + q0, 1024 + q1), (1024 + q2, 1024 + q3)
__device__ __forceinline__ void bytes_to_h2(uint32_t v, __half2& lo, __half2& hi) {
    lo = u2h(__byte_perm(v, 0x64u, 0x4140));
    hi = u2h(__byte_perm(v, 0x64u, 0x4342));
}

// 4 weights (bytes of v, value = (q - off) * s + m) -> b0, b1; off includes the 1024 of the conversion
__device__ __forceinline__ void w4(uint32_t v, __half2 off, __half2 s, __half2 m, uint32_t& b0, uint32_t& b1) {
    __half2 lo, hi;
    bytes_to_h2(v, lo, hi);
    b0 = h2u(__hfma2(__hsub2(lo, off), s, m));
    b1 = h2u(__hfma2(__hsub2(hi, off), s, m));
}

// the A fragment of lane's token for k .. k+3 (xr: the token's row of the block's X tile in shared memory)
__device__ __forceinline__ void xa(const __half* xr, int k, uint32_t& a0, uint32_t& a1) {
    const uint2 v = *(const uint2*) (xr + k);
    a0 = v.x;
    a1 = v.y;
}

__device__ __forceinline__ uint32_t wbyte(const uint32_t* w, int i) { return (w[i >> 2] >> (8 * (i & 3))) & 0xffu; }

__device__ __forceinline__ void scale_min_k4(int j, const uint32_t* w, float d, float dmin, float& sc, float& mn) {
    auto q = [&](int i) { return wbyte(w, 4 + i); };   // scales[12] at byte 4
    uint32_t sv, mv;
    if (j < 4) { sv = q(j) & 63; mv = q(j + 4) & 63; }
    else { sv = (q(j + 4) & 0xF) | ((q(j - 4) >> 6) << 4); mv = (q(j + 4) >> 4) | ((q(j) >> 6) << 4); }
    sc = d * (float) sv;
    mn = dmin * (float) mv;
}

// one super-block of the lane's row (words w) against the lane's token row xr (k from k0)
template <int TYPE> __device__ __forceinline__ void sb_mma(const uint32_t* w, int sh, const __half* xr, int k0, float* c) {
    const __half2 off1024 = __float2half2_rn(1024.0f);
    if constexpr (TYPE == 12 || TYPE == 13) {
        const float d = h2f(w[0] & 0xffff), dmin = h2f(w[0] >> 16);
        const int qs = TYPE == 12 ? 4 : 12;   // the qs words (Q5_K: after 32 bytes of qh)
#pragma unroll
        for (int j = 0; j < 4; ++j) {   // 64-value groups
            float d1, m1, d2, m2;
            scale_min_k4(2 * j, w, d, dmin, d1, m1);
            scale_min_k4(2 * j + 1, w, d, dmin, d2, m2);
            const __half2 s1 = __float2half2_rn(d1), s2 = __float2half2_rn(d2);
            const __half2 n1 = __float2half2_rn(-m1), n2 = __float2half2_rn(-m2);
#pragma unroll
            for (int c4 = 0; c4 < 8; ++c4) {   // 4 qs bytes: values 64j + 4 c4 .. +3 (low) and +32 (high)
                const uint32_t q = w[qs + 8 * j + c4];
                uint32_t lo = q & 0x0F0F0F0Fu, hi = (q >> 4) & 0x0F0F0F0Fu;
                if constexpr (TYPE == 13) {   // qh bit 2j (low) / 2j + 1 (high) of bytes 4 c4 .. +3
                    const uint32_t h = w[4 + c4];
                    lo |= ((h >> (2 * j)) & 0x01010101u) << 4;
                    hi |= ((h >> (2 * j + 1)) & 0x01010101u) << 4;
                }
                uint32_t a0, a1, b0, b1;
                w4(lo, off1024, s1, n1, b0, b1);
                xa(xr, k0 + 64 * j + 4 * c4, a0, a1);
                mma884(c, a0, a1, b0, b1);
                w4(hi, off1024, s2, n2, b0, b1);
                xa(xr, k0 + 64 * j + 32 + 4 * c4, a0, a1);
                mma884(c, a0, a1, b0, b1);
            }
        }
    } else if constexpr (TYPE == 14) {
        // ql[128] at 0, qh[64] at 128, scales[16] at 192, d at 208; the block starts sh (0 or 2) bytes into w[0]
        auto word = [&](int byte) -> uint32_t {   // 4 bytes from byte offset `byte` (a multiple of 4) of the block
            return sh == 0 ? w[byte >> 2] : __byte_perm(w[byte >> 2], w[(byte >> 2) + 1], 0x5432);
        };
        const uint32_t dw = word(208);
        const float d = h2f(dw & 0xffff);
        const __half2 off = __float2half2_rn(1024.0f + 32.0f), zero = __float2half2_rn(0.0f);
#pragma unroll
        for (int n = 0; n < 2; ++n) {   // 128 halves
            const uint32_t sw0 = word(192 + 8 * n), sw1 = word(196 + 8 * n);
            __half2 sc[8];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                sc[i] = __float2half2_rn(d * (float) (int8_t) ((sw0 >> (8 * i)) & 0xff));
                sc[4 + i] = __float2half2_rn(d * (float) (int8_t) ((sw1 >> (8 * i)) & 0xff));
            }
#pragma unroll
            for (int c4 = 0; c4 < 8; ++c4) {   // l = 4 c4 .. +3
                const int l = 4 * c4, is = l / 16;
                const uint32_t a = word(64 * n + l), a2 = word(64 * n + l + 32), h = word(128 + 32 * n + l);
                const uint32_t q1 = (a & 0x0F0F0F0Fu) | ((h << 4) & 0x30303030u);
                const uint32_t q2 = (a2 & 0x0F0F0F0Fu) | ((h << 2) & 0x30303030u);
                const uint32_t q3 = ((a >> 4) & 0x0F0F0F0Fu) | (h & 0x30303030u);
                const uint32_t q4 = ((a2 >> 4) & 0x0F0F0F0Fu) | ((h >> 2) & 0x30303030u);
                const int kb = k0 + 128 * n + l;
                uint32_t a0, a1, b0, b1;
                w4(q1, off, sc[is], zero, b0, b1);
                xa(xr, kb, a0, a1);
                mma884(c, a0, a1, b0, b1);
                w4(q2, off, sc[is + 2], zero, b0, b1);
                xa(xr, kb + 32, a0, a1);
                mma884(c, a0, a1, b0, b1);
                w4(q3, off, sc[is + 4], zero, b0, b1);
                xa(xr, kb + 64, a0, a1);
                mma884(c, a0, a1, b0, b1);
                w4(q4, off, sc[is + 6], zero, b0, b1);
                xa(xr, kb + 96, a0, a1);
                mma884(c, a0, a1, b0, b1);
            }
        }
    } else {   // Q8_0: 8 blocks of [half d | 32 int8], 34 bytes (2-byte aligned inside the 272)
        const __half2 off = __float2half2_rn(1024.0f + 128.0f), zero = __float2half2_rn(0.0f);
#pragma unroll
        for (int bb = 0; bb < 8; ++bb) {
            const int o = 34 * bb;   // even: the d half and the quants start at o, o + 2
            const float d = h2f((w[o >> 2] >> (8 * (o & 3))) & 0xffff);
            const __half2 s = __float2half2_rn(d);
#pragma unroll
            for (int c4 = 0; c4 < 8; ++c4) {
                const int b = o + 2 + 4 * c4;   // 2-byte aligned
                const uint32_t v = (b & 3) == 0 ? w[b >> 2] : __byte_perm(w[b >> 2], w[(b >> 2) + 1], 0x5432);
                uint32_t a0, a1, b0, b1;
                w4(v ^ 0x80808080u, off, s, zero, b0, b1);   // int8 -> q + 128 as a byte
                xa(xr, k0 + 32 * bb + 4 * c4, a0, a1);
                mma884(c, a0, a1, b0, b1);
            }
        }
    }
}

// a block's 128 rows of one super-block in shared memory: row r's words at r * stride (odd: lane reads conflict free),
// copied with coalesced 4-byte loads from the 4-aligned start of each row's block (Q6_K blocks may start 2 bytes in)
template <int TYPE> __host__ __device__ constexpr int row_words() { return TYPE == 14 ? 53 : Fmt<TYPE>::bytes / 4; }
template <int TYPE> __host__ __device__ constexpr int stride() { return row_words<TYPE>() | 1; }
constexpr int kXld = 256 + 8;   // halves: token rows 132 words apart, 8 tokens on distinct bank pairs

template <int TYPE>
__global__ void __launch_bounds__(kW * 32) mmt_kernel(const uint8_t* __restrict__ W, size_t row_bytes,
                                                      const __half* __restrict__ X, int K, int N, int M,
                                                      int sb_per_split, float* __restrict__ out) {
    constexpr int RW = row_words<TYPE>(), ST = stride<TYPE>(), NT = kW * 32, ROWS = kW * 32;
    constexpr int PR = (RW + 31) / 32;   // words of a row a lane carries
    __shared__ uint32_t sw[ROWS * ST];
    __shared__ __align__(16) __half sx[8 * kXld];
    const int tid = threadIdx.x, lane = tid & 31, wp = tid >> 5;
    const int nb0 = blockIdx.x * ROWS;
    const int nsb = K / 256;
    const int sb0 = blockIdx.y * sb_per_split, sb1 = min(nsb, sb0 + sb_per_split);
    const int tok = (lane & 3) + ((lane >> 4) << 2);
    float c[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t* mine = sw + wp * 32 * ST;   // the warp's 32 rows
    const int wrow0 = nb0 + wp * 32;
    // the warp's rows of super-block sb into registers: word wd = lane + 32 p of row r, 32 consecutive words a load
    // (coalesced); Q6_K blocks that start 2 bytes into a word are realigned here, so the dequant reads whole words
    uint32_t pf[32][PR];
    auto fetch = [&](int sb) {
#pragma unroll
        for (int r = 0; r < 32; ++r) {
            const uintptr_t a = (uintptr_t) (W + (size_t) min(wrow0 + r, N - 1) * row_bytes + (size_t) sb * Fmt<TYPE>::bytes);
            const uint32_t* q = (const uint32_t*) (a & ~(uintptr_t) 3);
#pragma unroll
            for (int p = 0; p < PR; ++p) {
                const int wd = lane + 32 * p;
                pf[r][p] = wd < RW ? __ldg(q + wd) : 0u;
            }
        }
    };
    if (sb0 < sb1) fetch(sb0);
    for (int sb = sb0; sb < sb1; ++sb) {
        __syncthreads();
        for (int i = tid; i < 8 * 32; i += NT) {   // X rows 0..7, k sb*256 .. +255
            const int r = i / 32, cc = (i % 32) * 8;
            *(uint4*) (sx + r * kXld + cc) = __ldg((const uint4*) (X + (size_t) r * K + (size_t) sb * 256 + cc));
        }
#pragma unroll
        for (int r = 0; r < 32; ++r) {
            // Q6_K: a block starting 2 bytes into a word is realigned with the next word (the next lane's)
            const bool mis = TYPE == 14 && (((uintptr_t) (W + (size_t) min(wrow0 + r, N - 1) * row_bytes +
                                                          (size_t) sb * Fmt<TYPE>::bytes)) & 3);
#pragma unroll
            for (int p = 0; p < PR; ++p) {
                uint32_t v = pf[r][p];
                if (TYPE == 14) {
                    uint32_t nx = __shfl_down_sync(0xffffffffu, v, 1);
                    if (p + 1 < PR) {
                        const uint32_t first = __shfl_sync(0xffffffffu, pf[r][p + 1 < PR ? p + 1 : p], 0);
                        if (lane == 31) nx = first;
                    }
                    if (mis) v = __byte_perm(v, nx, 0x5432);
                }
                if (lane + 32 * p < RW) mine[r * ST + lane + 32 * p] = v;
            }
        }
        __syncthreads();
        if (sb + 1 < sb1) fetch(sb + 1);
        sb_mma<TYPE>(mine + lane * ST, 0, sx + tok * kXld, 0, c);
    }
    const int n0 = nb0 + wp * 32;
    if (n0 >= N) return;
    // c -> out[(split * M + t) * N + n]
    const int j = lane & 3, hi = lane >> 4, b = 4 * ((lane >> 2) & 3);
    const int r0 = b + (j & 1) + 16 * hi, cc = b + (j & 2);
    const int rl[2] = {r0, r0 + 2}, cl[4] = {cc, cc + 1, cc + 16, cc + 17};
    const int ci[8][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int rlane = rl[ci[i][0]], clane = cl[ci[i][1]];
        const int t = (rlane & 3) + ((rlane >> 4) << 2), n = n0 + clane;
        if (t < M && n < N) out[((size_t) blockIdx.y * M + t) * N + n] = c[i];
    }
}

__global__ void reduce_kernel(const float* __restrict__ part, int splits, int64_t n, float* __restrict__ y) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = 0.0f;
    for (int k = 0; k < splits; ++k) s += part[(size_t) k * n + i];
    y[i] = s;
}

__global__ void x_half8_kernel(const float* __restrict__ x, int K, int M, __half* __restrict__ out) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) 8 * K) return;
    const int r = (int) (i / K);
    out[i] = r < M ? __float2half(x[i]) : __float2half(0.0f);
}

int sm_count() {
    static const int n = [] {
        int dev = 0, c = 80;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&c, cudaDevAttrMultiProcessorCount, dev);
        return c;
    }();
    return n;
}

}  // namespace

bool q35_mmt_supported(int ggml_type) { return ggml_type == 12 || ggml_type == 13 || ggml_type == 14 || ggml_type == 8; }

void q35_mmt_prep(const float* x, int K, int M, uint16_t* xh8, void* stream) {
    x_half8_kernel<<<(unsigned) (((int64_t) 8 * K + 255) / 256), 256, 0, (cudaStream_t) stream>>>(x, K, M, (__half*) xh8);
    check("q35_mmt_prep");
}

int64_t q35_mmt_part_floats(int N) { return (int64_t) 64 * 8 * N; }

void q35_mmt(int ggml_type, const void* W, const uint16_t* xh8, float* y, int K, int N, int M, float* part,
             void* stream) {
    if (M < 1 || M > 8 || K % 256 != 0) throw std::invalid_argument("q35_mmt: 1..8 rows, K a multiple of 256");
    int bytes = 0;
    switch (ggml_type) {
    case 12: bytes = 144; break;
    case 13: bytes = 176; break;
    case 14: bytes = 210; break;
    case 8: bytes = 272; break;
    default: throw std::invalid_argument("q35_mmt: weight type");
    }
    const size_t row_bytes = (size_t) K / 256 * bytes;
    const int nsb = K / 256, bx = (N + kW * 32 - 1) / (kW * 32);
    // k splits: enough blocks for 4 a multiprocessor, at least 2 super-blocks a split
    int splits = std::max(1, std::min(std::min(64, nsb / 2), (8 * sm_count() + bx - 1) / bx));
    const int per = (nsb + splits - 1) / splits;
    splits = (nsb + per - 1) / per;
    float* dst = splits == 1 ? y : part;
    if (splits > 1 && (int64_t) splits * M * N > q35_mmt_part_floats(N)) throw std::runtime_error("q35_mmt: partials");
    const dim3 grid(bx, splits);
    cudaStream_t s = (cudaStream_t) stream;
    const __half* X = (const __half*) xh8;
    const uint8_t* w = (const uint8_t*) W;
    switch (ggml_type) {
    case 12: mmt_kernel<12><<<grid, kW * 32, 0, s>>>(w, row_bytes, X, K, N, M, per, dst); break;
    case 13: mmt_kernel<13><<<grid, kW * 32, 0, s>>>(w, row_bytes, X, K, N, M, per, dst); break;
    case 14: mmt_kernel<14><<<grid, kW * 32, 0, s>>>(w, row_bytes, X, K, N, M, per, dst); break;
    case 8: mmt_kernel<8><<<grid, kW * 32, 0, s>>>(w, row_bytes, X, K, N, M, per, dst); break;
    }
    check("q35_mmt");
    if (splits > 1) {
        const int64_t n = (int64_t) M * N;
        reduce_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, s>>>(part, splits, n, y);
        check("q35_mmt_reduce");
    }
}

}  // namespace strata::kernels
