// Ternary types on CUDA: TQ1_0R_1024[_XK] dequantization and mat-vec, and the fp32-accumulated batched GEMM
// shared by all the ternary types (TQ1_0[_XK] and TQ1_0R_1024[_XK]). See tq1r.cuh for the arithmetic contract.
// ggml-cuda is compiled with -use_fast_math: the float operations of the int8 mat-vec epilogue are written with
// explicit round-to-nearest intrinsics in a fixed order, so that their result does not depend on contractions.

#include "tq1r.cuh"

#include <algorithm>
#include <cstdlib>

// ---------------------------------------------------------------------------------------------
// Contiguous dequantization (de-rotated weights), one 256-thread CUDA block per packed 1024-block.

template <int K, typename dst_t>
static __global__ void __launch_bounds__(256) k_dequant_tq1r(const void * __restrict__ vx, dst_t * __restrict__ y, const int64_t nblk) {
    __shared__ float sh[1024];
    const block_tq1_0r_1024 * x = (const block_tq1_0r_1024 *) vx;
    for (int64_t ib = blockIdx.x; ib < nblk; ib += gridDim.x) {
        tq1r_dequant_block<K>(x + ib*K, y + ib*QK_TQ1_0R_1024, sh);
    }
}

template <int K, typename dst_t>
static void dequantize_row_tq1r_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    GGML_ASSERT(k % QK_TQ1_0R_1024 == 0);
    const int64_t nblk = k / QK_TQ1_0R_1024;
    if (nblk == 0) {
        return;
    }
    const int grid = (int) MIN(nblk, (int64_t) 65535);
    k_dequant_tq1r<K, dst_t><<<grid, 256, 0, stream>>>(vx, y, nblk);
}

to_fp32_cuda_t ggml_cuda_tq1r_get_to_fp32(ggml_type type) {
    switch (type) {
        case GGML_TYPE_TQ1_0R_1024:    return dequantize_row_tq1r_cuda<1, float>;
        case GGML_TYPE_TQ1_0R_1024_X2: return dequantize_row_tq1r_cuda<2, float>;
        case GGML_TYPE_TQ1_0R_1024_X3: return dequantize_row_tq1r_cuda<3, float>;
        case GGML_TYPE_TQ1_0R_1024_X4: return dequantize_row_tq1r_cuda<4, float>;
        default:                       return nullptr;
    }
}

to_fp16_cuda_t ggml_cuda_tq1r_get_to_fp16(ggml_type type) {
    switch (type) {
        case GGML_TYPE_TQ1_0R_1024:    return dequantize_row_tq1r_cuda<1, half>;
        case GGML_TYPE_TQ1_0R_1024_X2: return dequantize_row_tq1r_cuda<2, half>;
        case GGML_TYPE_TQ1_0R_1024_X3: return dequantize_row_tq1r_cuda<3, half>;
        case GGML_TYPE_TQ1_0R_1024_X4: return dequantize_row_tq1r_cuda<4, half>;
        default:                       return nullptr;
    }
}

// ---------------------------------------------------------------------------------------------
// Activation rotation: y = (1/32) H_1024 (S . x) for every 1024-chunk of every row. ROTATE=false is a plain
// strided conversion (TQ1_0 family). Output is contiguous [ne10, ne11, ne12, ne13].

template <bool ROTATE, typename src_t, typename dst_t>
static __global__ void __launch_bounds__(256) k_rot_tq1r(
        const src_t * __restrict__ x, dst_t * __restrict__ y,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t nrows,
        const int64_t s11, const int64_t s12, const int64_t s13) {
    __shared__ float sh[1024];
    const int tid = threadIdx.x;
    const int64_t nchunk = ne10 / 1024;
    for (int64_t r = blockIdx.y; r < nrows; r += gridDim.y) {
        const int64_t i11 = r % ne11;
        const int64_t i12 = (r / ne11) % ne12;
        const int64_t i13 = r / (ne11*ne12);
        const src_t * xr = x + i11*s11 + i12*s12 + i13*s13;
        dst_t       * yr = y + r*ne10;
        for (int64_t c = blockIdx.x; c < nchunk; c += gridDim.x) {
            if constexpr (ROTATE) {
                for (int e = tid; e < 1024; e += 256) {
                    sh[e] = __fmul_rn(ggml_cuda_cast<float>(xr[c*1024 + e]), tq1r_sign(e) * 0.03125f);
                }
                __syncthreads();
                tq1r_fwht1024(sh, tid);
                for (int e = tid; e < 1024; e += 256) {
                    yr[c*1024 + e] = ggml_cuda_cast<dst_t>(sh[e]);
                }
                __syncthreads();
            } else {
                for (int e = tid; e < 1024; e += 256) {
                    yr[c*1024 + e] = ggml_cuda_cast<dst_t>(ggml_cuda_cast<float>(xr[c*1024 + e]));
                }
            }
        }
    }
}

template <bool ROTATE, typename src_t, typename dst_t>
static void tq1r_rot_launch(const src_t * x, dst_t * y, int64_t ne10, int64_t ne11, int64_t ne12, int64_t ne13,
                            int64_t s11, int64_t s12, int64_t s13, cudaStream_t stream) {
    const int64_t nrows = ne11*ne12*ne13;
    const dim3 grid((unsigned) MIN(ne10/1024, (int64_t) 65535), (unsigned) MIN(nrows, (int64_t) 65535), 1);
    k_rot_tq1r<ROTATE, src_t, dst_t><<<grid, 256, 0, stream>>>(x, y, ne10, ne11, ne12, nrows, s11, s12, s13);
}

void ggml_cuda_tq1r_rotate_f32(const float * x, float * y, int64_t ne10, int64_t ne11, int64_t ne12, int64_t ne13,
                               int64_t s11, int64_t s12, int64_t s13, cudaStream_t stream) {
    GGML_ASSERT(ne10 % 1024 == 0);
    tq1r_rot_launch<true, float, float>(x, y, ne10, ne11, ne12, ne13, s11, s12, s13, stream);
}

// Plain strided conversion for any ne10 (TQ1_0 family: rows are multiples of 256, not of 1024).
template <typename src_t, typename dst_t>
static __global__ void k_cvt_tq(const src_t * __restrict__ x, dst_t * __restrict__ y,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t nel,
        const int64_t s11, const int64_t s12, const int64_t s13) {
    for (int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x; i < nel; i += (int64_t) gridDim.x*blockDim.x) {
        const int64_t i10 = i % ne10;
        const int64_t r   = i / ne10;
        const int64_t i11 = r % ne11;
        const int64_t i12 = (r / ne11) % ne12;
        const int64_t i13 = r / (ne11*ne12);
        y[i] = ggml_cuda_cast<dst_t>(ggml_cuda_cast<float>(x[i10 + i11*s11 + i12*s12 + i13*s13]));
    }
}

template <typename src_t, typename dst_t>
static void tq_cvt_launch(const src_t * x, dst_t * y, int64_t ne10, int64_t ne11, int64_t ne12, int64_t ne13,
                          int64_t s11, int64_t s12, int64_t s13, cudaStream_t stream) {
    const int64_t nel = ne10*ne11*ne12*ne13;
    const int64_t grid = MIN((nel + 255)/256, (int64_t) 65535*8);
    k_cvt_tq<src_t, dst_t><<<(unsigned) grid, 256, 0, stream>>>(x, y, ne10, ne11, ne12, nel, s11, s12, s13);
}

// ---------------------------------------------------------------------------------------------
// Rotation + int8 quantization in the BYTE order of the TQ1_0R_1024 weight. One scale per quarter (256
// elements): amax/127. Output per row and per 1024-block: TQ1R_YQ_WORDS uint32
//   [0, 208)   Y4[s]   slot s = word ws*4 + byte k: int8 of digits t = 0..3 of that byte
//   [208, 260) Y4b[ws] int8 of digit t = 4 of the 4 bytes of word ws (zero for qh words)
//   [260, 264) float bits of the quarter scale
// The packed planes of a block share this buffer: they are laid out exactly like the base block.

#define TQ1R_YQ_WORDS 264
#define TQ1R_W_WORDS   56

static __device__ __forceinline__ uint32_t tq1r_q8(const float v, const float inv) {
    int q = __float2int_rn(__fmul_rn(v, inv));
    q = q < -127 ? -127 : (q > 127 ? 127 : q);
    return (uint32_t) q & 255u;
}

template <typename src_t>
static __global__ void __launch_bounds__(256) k_rot_q8_tq1r(
        const src_t * __restrict__ x, uint32_t * __restrict__ yq,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t nrows,
        const int64_t s11, const int64_t s12, const int64_t s13) {
    __shared__ float sh[1024];
    __shared__ float red[256];
    const int tid = threadIdx.x;
    const int64_t nchunk = ne10 / 1024;
    for (int64_t r = blockIdx.y; r < nrows; r += gridDim.y) {
        const int64_t i11 = r % ne11;
        const int64_t i12 = (r / ne11) % ne12;
        const int64_t i13 = r / (ne11*ne12);
        const src_t * xr = x + i11*s11 + i12*s12 + i13*s13;
        for (int64_t c = blockIdx.x; c < nchunk; c += gridDim.x) {
            for (int e = tid; e < 1024; e += 256) {
                sh[e] = __fmul_rn(ggml_cuda_cast<float>(xr[c*1024 + e]), tq1r_sign(e) * 0.03125f);
            }
            __syncthreads();
            tq1r_fwht1024(sh, tid);

            // per-quarter absolute maximum (exact in any order)
            red[tid] = fmaxf(fmaxf(fabsf(sh[4*tid]), fabsf(sh[4*tid + 1])), fmaxf(fabsf(sh[4*tid + 2]), fabsf(sh[4*tid + 3])));
            __syncthreads();
            for (int st = 32; st > 0; st >>= 1) {
                if ((tid & 63) < st) {
                    red[tid] = fmaxf(red[tid], red[tid + st]);
                }
                __syncthreads();
            }

            uint32_t * ob = yq + (r*nchunk + c)*TQ1R_YQ_WORDS;
            if (tid < 208) {
                const int ws = tid >> 2, k = tid & 3;
                const int q = ws / 13, rr = ws - q*13;
                int e0, es;
                if (rr < 8)       { e0 = q*256 + 4*rr + k;             es = 32; }
                else if (rr < 12) { e0 = q*256 + 160 + 4*(rr - 8) + k; es = 16; }
                else              { e0 = q*256 + 240 + k;              es = 4;  }
                const float amax = red[q*64];
                const float inv  = amax > 0.0f ? __fdiv_rn(127.0f, amax) : 0.0f;
                uint32_t pk = 0;
#pragma unroll
                for (int t = 0; t < 4; ++t) {
                    pk |= tq1r_q8(sh[e0 + t*es], inv) << (8*t);
                }
                ob[tid] = pk;
            }
            if (tid < 52) {
                const int ws = tid;
                const int q = ws / 13, rr = ws - q*13;
                const float amax = red[q*64];
                const float inv  = amax > 0.0f ? __fdiv_rn(127.0f, amax) : 0.0f;
                uint32_t pk = 0;
                if (rr < 12) {
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        const int e = rr < 8 ? (q*256 + 4*rr + k + 4*32) : (q*256 + 160 + 4*(rr - 8) + k + 4*16);
                        pk |= tq1r_q8(sh[e], inv) << (8*k);
                    }
                }
                ob[208 + ws] = pk;
            }
            if (tid < 4) {
                ob[260 + tid] = __float_as_uint(__fdiv_rn(red[tid*64], 127.0f));
            }
            __syncthreads(); // sh and red are reused by the next chunk
        }
    }
}

template <typename src_t>
static void tq1r_rot_q8_launch(const src_t * x, uint32_t * yq, int64_t ne10, int64_t ne11, int64_t ne12, int64_t ne13,
                               int64_t s11, int64_t s12, int64_t s13, cudaStream_t stream) {
    const int64_t nrows = ne11*ne12*ne13;
    const dim3 grid((unsigned) MIN(ne10/1024, (int64_t) 65535), (unsigned) MIN(nrows, (int64_t) 65535), 1);
    k_rot_q8_tq1r<src_t><<<grid, 256, 0, stream>>>(x, yq, ne10, ne11, ne12, nrows, s11, s12, s13);
}

// ---------------------------------------------------------------------------------------------
// int8 mat-vec. One warp per output row; lane l handles the weight words f = l, l+32, ... (f = blk*52 + ws)
// of every plane. For one word the K planes are dotted with the same activation words and summed, then the
// sum is scaled by the quarter scale of the activation. With K = 1 this is the single-plane kernel.

struct tq1r_mmv_params {
    const char     * w;      // src0 data
    const uint32_t * yq;     // rotated + quantized activation, TQ1R_YQ_WORDS per (row of src1, block)
    float          * dst;
    const int32_t  * ids;    // MUL_MAT_ID only
    int64_t nb;              // 1024-blocks per weight row
    int64_t nrows;           // ne01
    int64_t nb01, nb02, nb03;
    int64_t ne11, ne12;      // src1 dims (rows of the quantized activation buffer)
    int64_t nb1, nb2, nb3;   // dst strides (bytes)
    int64_t r2, r3;          // broadcast factors (MUL_MAT)
    int64_t ids_s0, ids_s1;  // ids strides (elements)
};

static __device__ __forceinline__ uint32_t tq1r_digits4(const uint32_t w, const int t) {
    return  (uint32_t) tq1_digit( w        & 255u, t)        |
           ((uint32_t) tq1_digit((w >>  8) & 255u, t) <<  8) |
           ((uint32_t) tq1_digit((w >> 16) & 255u, t) << 16) |
           ((uint32_t) tq1_digit( w >> 24        , t) << 24);
}

// digits 0..3 of byte b as 4 int8
static __device__ __forceinline__ int tq1r_lut_a(const uint32_t b) {
    return (int) (tq1_digit(b, 0) | (tq1_digit(b, 1) << 8) | (tq1_digit(b, 2) << 16) | (tq1_digit(b, 3) << 24));
}

template <int ncols, int K>
static __global__ void __launch_bounds__(128) k_mmv_tq1r_q8(const tq1r_mmv_params p) {
    const int lane = threadIdx.x;
    const int64_t row = (int64_t) blockIdx.x*blockDim.y + threadIdx.y;
    if (row >= p.nrows) {
        return;
    }

    const char * wrow;
    int64_t ycol0;
    float * drow;
    if (p.ids) {
        // MUL_MAT_ID, ncols == 1: blockIdx.y = expert slot, blockIdx.z = token
        const int64_t e = blockIdx.y;
        const int64_t t = blockIdx.z;
        const int32_t expert = p.ids[t*p.ids_s1 + e*p.ids_s0];
        drow  = (float *) ((char *) p.dst + t*p.nb2 + e*p.nb1) + row;
        if (expert < 0) { // skipped slot: zero, as on the CPU backend
            if (lane == 0) {
                *drow = 0.0f;
            }
            return;
        }
        wrow  = p.w + expert*p.nb02 + row*p.nb01;
        ycol0 = t*p.ne11 + (e % p.ne11);
    } else {
        const int64_t i2 = blockIdx.y;
        const int64_t i3 = blockIdx.z;
        wrow  = p.w + (i3/p.r3)*p.nb03 + (i2/p.r2)*p.nb02 + row*p.nb01;
        ycol0 = (i3*p.ne12 + i2)*p.ne11;
        drow  = (float *) ((char *) p.dst + i3*p.nb3 + i2*p.nb2) + row;
    }
    const uint32_t * W = (const uint32_t *) wrow;

    float temp[ncols];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        temp[j] = 0.0f;
    }

    const int ones = 0x01010101;
    const int64_t nws = p.nb*52;
    for (int64_t f = lane; f < nws; f += WARP_SIZE) {
        const int64_t blk = f / 52;
        const int ws = (int) (f - blk*52);
        const int q  = ws / 13;
        const int r  = ws - q*13;
        const bool c1 = r < 8;
        const int wword = r < 12 ? (q*12 + r) : (48 + q);

        float acc[ncols] = {0.0f};
#pragma unroll
        for (int pl = 0; pl < K; ++pl) {
            const uint32_t * Wb = W + (blk*K + pl)*TQ1R_W_WORDS;
            const uint32_t w  = Wb[wword];
            const uint32_t dw = Wb[52 + q];
            const float d0 = __half2float(__ushort_as_half((unsigned short) (dw & 0xFFFFu)));
            const float d1 = __half2float(__ushort_as_half((unsigned short) (dw >> 16)));

            const uint32_t b0 = w & 255u, b1 = (w >> 8) & 255u, b2 = (w >> 16) & 255u, b3 = w >> 24;
            const int la0 = tq1r_lut_a(b0), la1 = tq1r_lut_a(b1), la2 = tq1r_lut_a(b2), la3 = tq1r_lut_a(b3);
            const int lb  = (int) tq1r_digits4(w, 4);

#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                const uint32_t * yb = p.yq + ((ycol0 + j)*p.nb + blk)*TQ1R_YQ_WORDS;
                const int Y0 = (int) yb[4*ws + 0];
                const int Y1 = (int) yb[4*ws + 1];
                const int Y2 = (int) yb[4*ws + 2];
                const int Y3 = (int) yb[4*ws + 3];
                const int Yb = (int) yb[208 + ws];
                const int sY  = ggml_cuda_dp4a(Y3, ones, ggml_cuda_dp4a(Y2, ones, ggml_cuda_dp4a(Y1, ones, ggml_cuda_dp4a(Y0, ones, 0))));
                const int sYb = ggml_cuda_dp4a(Yb, ones, 0);
                const int ilo = ggml_cuda_dp4a(la3, Y3, ggml_cuda_dp4a(la2, Y2, ggml_cuda_dp4a(la1, Y1, ggml_cuda_dp4a(la0, Y0, 0)))) - sY;
                const int ihi = ggml_cuda_dp4a(lb, Yb, 0) - sYb;
                const float pa = c1 ? __fadd_rn(__fmul_rn(d0, (float) ilo), __fmul_rn(d1, (float) ihi))
                                    : __fmul_rn(d1, (float) (ilo + ihi));
                acc[j] = pl == 0 ? pa : __fadd_rn(acc[j], pa);
            }
        }

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            const float ys = __uint_as_float(p.yq[((ycol0 + j)*p.nb + blk)*TQ1R_YQ_WORDS + 260 + q]);
            temp[j] = __fadd_rn(temp[j], __fmul_rn(ys, acc[j]));
        }
    }

#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        float v = temp[j];
#pragma unroll
        for (int off = WARP_SIZE/2; off > 0; off >>= 1) {
            v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFF, v, off, WARP_SIZE));
        }
        if (lane == 0) {
            *(float *) ((char *) drow + j*p.nb1) = v;
        }
    }
}

template <int K>
static void tq1r_mmv_launch_k(const tq1r_mmv_params & p, const int ncols, const dim3 grid, cudaStream_t stream) {
    const dim3 block(WARP_SIZE, 4, 1);
    switch (ncols) {
        case 1: k_mmv_tq1r_q8<1, K><<<grid, block, 0, stream>>>(p); break;
        case 2: k_mmv_tq1r_q8<2, K><<<grid, block, 0, stream>>>(p); break;
        case 3: k_mmv_tq1r_q8<3, K><<<grid, block, 0, stream>>>(p); break;
        case 4: k_mmv_tq1r_q8<4, K><<<grid, block, 0, stream>>>(p); break;
        case 5: k_mmv_tq1r_q8<5, K><<<grid, block, 0, stream>>>(p); break;
        case 6: k_mmv_tq1r_q8<6, K><<<grid, block, 0, stream>>>(p); break;
        case 7: k_mmv_tq1r_q8<7, K><<<grid, block, 0, stream>>>(p); break;
        case 8: k_mmv_tq1r_q8<8, K><<<grid, block, 0, stream>>>(p); break;
        default: GGML_ABORT("tq1r mmv: unsupported ncols %d", ncols);
    }
}

static void tq1r_mmv_launch(const tq1r_mmv_params & p, const int K, const int ncols, const dim3 grid, cudaStream_t stream) {
    switch (K) {
        case 1: tq1r_mmv_launch_k<1>(p, ncols, grid, stream); break;
        case 2: tq1r_mmv_launch_k<2>(p, ncols, grid, stream); break;
        case 3: tq1r_mmv_launch_k<3>(p, ncols, grid, stream); break;
        case 4: tq1r_mmv_launch_k<4>(p, ncols, grid, stream); break;
        default: GGML_ABORT("tq1r mmv: unsupported number of planes %d", K);
    }
}

// ---------------------------------------------------------------------------------------------
// Plain ternary decode (trit x d, no rotation) of nrows weight rows starting at x into a plane-major
// [K][nrows][ne00] buffer. Exact in fp16: d is fp16 and trit is in {-1, 0, +1}.

template <bool R1024, int K, typename dst_t>
static __global__ void k_dequant_plain_tq(const char * __restrict__ x, dst_t * __restrict__ y,
        const int64_t ne00, const int64_t nb01, const int64_t nel_plane) {
    const int64_t nel = nel_plane*K;
    for (int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x; i < nel; i += (int64_t) gridDim.x*blockDim.x) {
        const int64_t pl  = i / nel_plane;
        const int64_t j   = i - pl*nel_plane;
        const int64_t r   = j / ne00;
        const int64_t i00 = j - r*ne00;
        const char * xr = x + r*nb01;
        float v;
        if constexpr (R1024) {
            const block_tq1_0r_1024 * b = (const block_tq1_0r_1024 *) xr + (i00/QK_TQ1_0R_1024)*K + pl;
            const int e = (int) (i00 % QK_TQ1_0R_1024);
            const int q = e >> 8;
            v = (float) tq1_trit(b->qs[q], b->qh[q], e & 255) * __half2float(b->d[e >> 7]);
        } else {
            const block_tq1_0 * b = (const block_tq1_0 *) xr + (i00/QK_K)*K + pl;
            v = (float) tq1_trit(b->qs, b->qh, (int) (i00 % QK_K)) * __half2float(b->d);
        }
        y[i] = ggml_cuda_cast<dst_t>(v);
    }
}

template <bool R1024, int K, typename dst_t>
static void tq_dequant_plain_launch_k(const char * x, dst_t * y, const int64_t ne00, const int64_t nrows, const int64_t nb01, cudaStream_t stream) {
    const int64_t nel_plane = ne00*nrows;
    const int64_t grid = MIN((nel_plane*K + 255)/256, (int64_t) 65535*8);
    k_dequant_plain_tq<R1024, K, dst_t><<<(unsigned) grid, 256, 0, stream>>>(x, y, ne00, nb01, nel_plane);
}

template <typename dst_t>
static void tq_dequant_plain_launch(const ggml_type type, const char * x, dst_t * y, const int64_t ne00, const int64_t nrows,
                                    const int64_t nb01, cudaStream_t stream) {
    switch (type) {
        case GGML_TYPE_TQ1_0:          tq_dequant_plain_launch_k<false, 1>(x, y, ne00, nrows, nb01, stream); break;
        case GGML_TYPE_TQ1_0_X2:       tq_dequant_plain_launch_k<false, 2>(x, y, ne00, nrows, nb01, stream); break;
        case GGML_TYPE_TQ1_0_X3:       tq_dequant_plain_launch_k<false, 3>(x, y, ne00, nrows, nb01, stream); break;
        case GGML_TYPE_TQ1_0_X4:       tq_dequant_plain_launch_k<false, 4>(x, y, ne00, nrows, nb01, stream); break;
        case GGML_TYPE_TQ1_0R_1024:    tq_dequant_plain_launch_k<true,  1>(x, y, ne00, nrows, nb01, stream); break;
        case GGML_TYPE_TQ1_0R_1024_X2: tq_dequant_plain_launch_k<true,  2>(x, y, ne00, nrows, nb01, stream); break;
        case GGML_TYPE_TQ1_0R_1024_X3: tq_dequant_plain_launch_k<true,  3>(x, y, ne00, nrows, nb01, stream); break;
        case GGML_TYPE_TQ1_0R_1024_X4: tq_dequant_plain_launch_k<true,  4>(x, y, ne00, nrows, nb01, stream); break;
        default: GGML_ABORT("tq gemm: unsupported type %s", ggml_type_name(type));
    }
}

static bool tq1r_batch_f32() {
    static const bool on = []() { const char * e = getenv("GGML_CUDA_TQ1R_F32"); return e != nullptr && atoi(e) != 0; }();
    return on;
}

// decoded weight rows per GEMM tile: bounds the temporary buffer (e.g. a 248k-row output head)
static constexpr int64_t TQ_GEMM_TILE_BYTES = 64ll << 20;

// Batched path: rotated (TQ1_0R_1024 family) or plain (TQ1_0 family) activation x' and plain-decoded weights.
// The weights of every matrix are decoded tile by tile (rows), plane by plane, and each plane is one cuBLAS GEMM
// with fp16 operands and fp32 accumulation into dst (beta = 1 from the second plane on), so every plane enters
// the product exactly, as with separate plane tensors.
void ggml_cuda_mul_mat_tq_batch(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS

    const ggml_type type = src0->type;
    const int  K      = ggml_cuda_tq_planes(type);
    const bool rotate = ggml_cuda_tq_is_r1024(type);

    GGML_ASSERT(K > 0);
    GGML_ASSERT(src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ne10 == ne00);
    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(ne12 % ne02 == 0 && ne13 % ne03 == 0);
    GGML_ASSERT(!rotate || ne10 % 1024 == 0);

    cudaStream_t stream = ctx.stream();
    cublasHandle_t handle = ctx.cublas_handle();

    const bool f32 = tq1r_batch_f32();
    const size_t ts = f32 ? sizeof(float) : sizeof(half);

    const int64_t ts1 = ggml_type_size(src1->type);
    const int64_t s11 = nb11/ts1, s12 = nb12/ts1, s13 = nb13/ts1;
    const int64_t nrows1 = ne11*ne12*ne13;

    // activation: rotated per 1024-chunk (TQ1_0R_1024 family) or plain strided conversion (TQ1_0 family)
    ggml_cuda_pool_alloc<char> xbuf(ctx.pool(), ne10*nrows1*ts);
#define TQ_ROT(ST, DT) tq1r_rot_launch<true, ST, DT>((const ST *) src1->data, (DT *) xbuf.get(), ne10, ne11, ne12, ne13, s11, s12, s13, stream)
#define TQ_CVT(ST, DT) tq_cvt_launch<ST, DT>((const ST *) src1->data, (DT *) xbuf.get(), ne10, ne11, ne12, ne13, s11, s12, s13, stream)
    if (src1->type == GGML_TYPE_F32) {
        if (rotate) { if (f32) { TQ_ROT(float, float); } else { TQ_ROT(float, half); } }
        else        { if (f32) { TQ_CVT(float, float); } else { TQ_CVT(float, half); } }
    } else {
        if (rotate) { if (f32) { TQ_ROT(half, float); } else { TQ_ROT(half, half); } }
        else        { if (f32) { TQ_CVT(half, float); } else { TQ_CVT(half, half); } }
    }
#undef TQ_CVT
#undef TQ_ROT
    CUDA_CHECK(cudaGetLastError());

    const int64_t row_bytes = (int64_t) K*ne00*ts;
    const int64_t tile      = std::max<int64_t>(1, std::min<int64_t>(ne01, TQ_GEMM_TILE_BYTES/row_bytes));
    ggml_cuda_pool_alloc<char> wbuf(ctx.pool(), tile*row_bytes);

    const float alpha    = 1.0f;
    const float beta_set = 0.0f;
    const float beta_add = 1.0f;
    const int64_t r2 = ne12/ne02;
    const int64_t r3 = ne13/ne03;
    const cudaDataType_t dt = f32 ? CUDA_R_32F : CUDA_R_16F;
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    // the handle may allow TF32: the verification path must stay fp32
    const cublasComputeType_t ct = f32 ? CUBLAS_COMPUTE_32F_PEDANTIC : CUBLAS_COMPUTE_32F;
#else
    const cublasComputeType_t ct = CUBLAS_COMPUTE_32F;
#endif

    for (int64_t i03 = 0; i03 < ne03; ++i03) {
        for (int64_t i02 = 0; i02 < ne02; ++i02) {
            const char * x = (const char *) src0->data + i02*nb02 + i03*nb03;
            for (int64_t row0 = 0; row0 < ne01; row0 += tile) {
                const int64_t nr = std::min<int64_t>(tile, ne01 - row0);
                if (f32) {
                    tq_dequant_plain_launch<float>(type, x + row0*nb01, (float *) wbuf.get(), ne00, nr, nb01, stream);
                } else {
                    tq_dequant_plain_launch<half >(type, x + row0*nb01, (half  *) wbuf.get(), ne00, nr, nb01, stream);
                }
                CUDA_CHECK(cudaGetLastError());

                for (int64_t i13 = i03*r3; i13 < (i03 + 1)*r3; ++i13) {
                    for (int64_t i12 = i02*r2; i12 < (i02 + 1)*r2; ++i12) {
                        const char * B = xbuf.get() + (i13*ne12 + i12)*ne11*ne10*ts;
                        float      * C = (float *) ((char *) dst->data + i12*nb2 + i13*nb3) + row0;
                        for (int pl = 0; pl < K; ++pl) {
                            const char * A = wbuf.get() + pl*nr*ne00*ts;
                            CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                nr, ne11, ne10,
                                &alpha, A, dt, ne00,
                                        B, dt, ne10,
                                pl == 0 ? &beta_set : &beta_add,
                                        C, CUDA_R_32F, nb1/sizeof(float),
                                ct, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------

void ggml_cuda_mul_mat_tq1r(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                            const ggml_tensor * ids, ggml_tensor * dst, bool force_batch) {
    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT(ggml_cuda_tq_is_r1024(src0->type));
    GGML_ASSERT(src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ne00 % QK_TQ1_0R_1024 == 0);
    GGML_ASSERT(ne10 == ne00);
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));
    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    // the int8 kernel reads the weight as 32-bit words
    GGML_ASSERT(nb01 % 4 == 0 && nb02 % 4 == 0 && nb03 % 4 == 0 && ((uintptr_t) src0->data) % 4 == 0);

    cudaStream_t stream = ctx.stream();

    const bool mmv = ids ? ne2 <= TQ1R_MMID_MAX_BATCH : (!force_batch && ne11 <= TQ1R_MMV_MAX_BATCH);
    if (!mmv) {
        GGML_ASSERT(ids == nullptr);
        ggml_cuda_mul_mat_tq_batch(ctx, src0, src1, dst);
        return;
    }

    const int K = ggml_cuda_tq_planes(src0->type);
    const int64_t nb = ne00 / QK_TQ1_0R_1024;
    const int64_t ts1 = ggml_type_size(src1->type);
    const int64_t s11 = nb11/ts1, s12 = nb12/ts1, s13 = nb13/ts1;
    const int64_t nrows1 = ne11*ne12*ne13;

    ggml_cuda_pool_alloc<uint32_t> yq(ctx.pool(), nrows1*nb*TQ1R_YQ_WORDS);
    if (src1->type == GGML_TYPE_F32) {
        tq1r_rot_q8_launch<float>((const float *) src1->data, yq.get(), ne10, ne11, ne12, ne13, s11, s12, s13, stream);
    } else {
        tq1r_rot_q8_launch<half>((const half *) src1->data, yq.get(), ne10, ne11, ne12, ne13, s11, s12, s13, stream);
    }
    CUDA_CHECK(cudaGetLastError());

    tq1r_mmv_params p;
    p.w     = (const char *) src0->data;
    p.yq    = yq.get();
    p.dst   = (float *) dst->data;
    p.ids   = ids ? (const int32_t *) ids->data : nullptr;
    p.nb    = nb;
    p.nrows = ne01;
    p.nb01  = nb01; p.nb02 = nb02; p.nb03 = nb03;
    p.ne11  = ne11; p.ne12 = ne12;
    p.nb1   = nb1;  p.nb2  = nb2;  p.nb3  = nb3;
    p.r2    = ids ? 1 : ne12/ne02;
    p.r3    = ids ? 1 : ne13/ne03;
    p.ids_s0 = ids ? ids->nb[0]/ggml_type_size(ids->type) : 0;
    p.ids_s1 = ids ? ids->nb[1]/ggml_type_size(ids->type) : 0;

    const unsigned gx = (unsigned) ((ne01 + 3) / 4);
    if (ids) {
        GGML_ASSERT(ids->type == GGML_TYPE_I32);
        // dst: [ne0, n_expert_used, n_tokens]
        tq1r_mmv_launch(p, K, 1, dim3(gx, (unsigned) ne1, (unsigned) ne2), stream);
    } else {
        GGML_ASSERT(ne12 % ne02 == 0 && ne13 % ne03 == 0);
        tq1r_mmv_launch(p, K, (int) ne11, dim3(gx, (unsigned) ne12, (unsigned) ne13), stream);
    }
    CUDA_CHECK(cudaGetLastError());
}
