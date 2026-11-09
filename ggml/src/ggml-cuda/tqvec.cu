// Ternary types on CUDA: batch-1..8 mat-vec (MUL_MAT ne11 <= 8, MUL_MAT_ID n_tokens <= 8) for both families,
// TQ1_0[_XK] and TQ1_0R_1024[_XK], one kernel.
//
// Activation: int8 in natural element order with one fp32 scale and one int32 sum per 128 elements, rotated per
// 1024-chunk first for the TQ1_0R_1024 family (y = (1/32) H_1024 (S x), as in tq1r.cu).
//
// Weight: both families are groups of 256 trits laid out like block_tq1_0 (qs[48]: 5 digits per byte, qh[4]: 4
// digits per byte). Unit of work = one 32-bit word of a group (12 qs words + 1 qh word). For digit t the 4 bytes
// of a word cover 4 consecutive elements, i.e. one int8x4 word of the activation:
//   qs word r < 8  : element 32t + 4r      (t = 0..4)  digits 0..3 in the first 128, digit 4 in the second
//   qs word r 8..11: element 160 + 16t + 4(r-8)        all in the second 128
//   qh word        : element 240 + 4t      (t = 0..3)  all in the second 128
// The digits are decoded in registers (SWAR Horner, bytes 0,2 and 1,3 in 16-bit fields: c_(t+1) = 3 c_t mod 256,
// digit_t = (3 c_t) >> 8). trit = digit - 1: the int8 sum of each 128 of the activation is subtracted once, by the
// lane that handles word 0 of the group. The K packed planes of a word share the activation words.
//
// Thread mapping: one warp = R consecutive rows, lanes stride over the words of the row; the activation words of
// a lane are loaded once and reused by the R rows, the K planes and (ncols > 1) the decode by the ncols columns.

#include "tq1r.cuh"

#include <cstdlib>

bool ggml_cuda_tq_vec_enabled() {
    static const bool on = []() { const char * e = getenv("GGML_CUDA_TQ_VEC"); return e == nullptr || atoi(e) != 0; }();
    return on;
}

static int tqvec_env_int(const char * name, int def) {
    const char * e = getenv(name);
    return e ? atoi(e) : def;
}

// ---------------------------------------------------------------------------------------------
// Activation quantization. One 256-thread CUDA block per (1024-chunk, row); warp w owns elements
// [128w, 128w + 128) of the chunk, lane l elements 128w + 4l .. 128w + 4l + 3.

template <bool ROTATE, typename src_t>
static __global__ void __launch_bounds__(256) k_tqvec_q8(
        const src_t * __restrict__ x, uint32_t * __restrict__ yw, float * __restrict__ ys, int * __restrict__ yi,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t nrows,
        const int64_t s11, const int64_t s12, const int64_t s13) {
    __shared__ float sh[ROTATE ? 1024 : 1];
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int64_t nchunk = (ne10 + 1023) / 1024;
    for (int64_t r = blockIdx.y; r < nrows; r += gridDim.y) {
        const int64_t i11 = r % ne11;
        const int64_t i12 = (r / ne11) % ne12;
        const int64_t i13 = r / (ne11*ne12);
        const src_t * xr = x + i11*s11 + i12*s12 + i13*s13;
        for (int64_t c = blockIdx.x; c < nchunk; c += gridDim.x) {
            const int64_t e0 = c*1024 + warp*128 + lane*4;
            float v[4];
            if constexpr (ROTATE) {
                for (int e = tid; e < 1024; e += 256) {
                    sh[e] = __fmul_rn(ggml_cuda_cast<float>(xr[c*1024 + e]), tq1r_sign(e) * 0.03125f);
                }
                __syncthreads();
                tq1r_fwht1024(sh, tid);
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    v[k] = sh[warp*128 + lane*4 + k];
                }
                __syncthreads(); // sh is reused by the next chunk
            } else {
                if (c*1024 + warp*128 >= ne10) { // ne10 % 128 == 0: warp-uniform
                    continue;
                }
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    v[k] = ggml_cuda_cast<float>(xr[e0 + k]);
                }
            }

            float amax = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, off, 32));
            }
            const float inv = amax > 0.0f ? __fdiv_rn(127.0f, amax) : 0.0f;
            uint32_t pk = 0;
            int isum = 0;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                int q = __float2int_rn(__fmul_rn(v[k], inv));
                q = q < -127 ? -127 : (q > 127 ? 127 : q);
                isum += q;
                pk |= ((uint32_t) q & 255u) << (8*k);
            }
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                isum += __shfl_xor_sync(0xFFFFFFFF, isum, off, 32);
            }
            yw[r*(ne10/4) + e0/4] = pk;
            if (lane == 0) {
                ys[r*(ne10/128) + e0/128] = __fdiv_rn(amax, 127.0f);
                yi[r*(ne10/128) + e0/128] = isum;
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Mat-vec

struct tqvec_params {
    const char     * w;      // src0 data
    const uint32_t * yw;     // int8x4 activation, nq*64 words per activation row
    const float    * ys;     // scale per 128, nq*2 per activation row
    const int      * yi;     // int8 sum per 128, nq*2 per activation row
    float          * dst;
    const int32_t  * ids;    // MUL_MAT_ID only
    int     nq;              // groups of 256 per weight row
    int64_t nrows;           // ne01
    int64_t nb01, nb02, nb03;
    int64_t ne11, ne12;
    int64_t nb1, nb2, nb3;   // dst strides (bytes)
    int64_t r2, r3;          // broadcast factors (MUL_MAT)
    int64_t ids_s0, ids_s1;  // ids strides (elements)
};

static __device__ __forceinline__ uint32_t tqvec_load_b2(const char * p) { // 2-byte aligned 32-bit load
    const uint16_t * p16 = (const uint16_t *) p;
    return (uint32_t) p16[0] | ((uint32_t) p16[1] << 16);
}

// 5 base-3 digits (int8x4 each, digit t of the 4 bytes) of a weight word
static __device__ __forceinline__ void tqvec_decode(const uint32_t w, int D[5]) {
    uint32_t a = __byte_perm(w, 0, 0x4240); // bytes 0, 2 in the two 16-bit fields
    uint32_t b = __byte_perm(w, 0, 0x4341); // bytes 1, 3
#pragma unroll
    for (int t = 0; t < 5; ++t) {
        const uint32_t ma = a*3u;
        const uint32_t mb = b*3u;
        D[t] = (int) __byte_perm(ma, mb, 0x7351); // bits 8..15 of the 4 fields = the 4 digits
        a = ma & 0x00FF00FFu;
        b = mb & 0x00FF00FFu;
    }
}

template <bool R1024, int K, int ncols, int R>
static __global__ void __launch_bounds__(256) k_tqvec(const tqvec_params p) {
    const int lane = threadIdx.x;
    const int64_t row0 = ((int64_t) blockIdx.x*blockDim.y + threadIdx.y)*R;
    if (row0 >= p.nrows) {
        return;
    }
    const int nr = (int) min((int64_t) R, p.nrows - row0);

    const char * wrow;
    int64_t ycol0;
    float * drow;
    if (p.ids) {
        // MUL_MAT_ID, ncols == 1: blockIdx.y = expert slot, blockIdx.z = token
        const int64_t e = blockIdx.y;
        const int64_t t = blockIdx.z;
        const int32_t expert = p.ids[t*p.ids_s1 + e*p.ids_s0];
        drow = (float *) ((char *) p.dst + t*p.nb2 + e*p.nb1) + row0;
        if (expert < 0) { // skipped slot: zero, as on the CPU backend
            if (lane < nr) {
                drow[lane] = 0.0f;
            }
            return;
        }
        wrow  = p.w + expert*p.nb02 + row0*p.nb01;
        ycol0 = t*p.ne11 + (e % p.ne11);
    } else {
        const int64_t i2 = blockIdx.y;
        const int64_t i3 = blockIdx.z;
        wrow  = p.w + (i3/p.r3)*p.nb03 + (i2/p.r2)*p.nb02 + row0*p.nb01;
        ycol0 = (i3*p.ne12 + i2)*p.ne11;
        drow  = (float *) ((char *) p.dst + i3*p.nb3 + i2*p.nb2) + row0;
    }

    float temp[ncols][R];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
#pragma unroll
        for (int n = 0; n < R; ++n) {
            temp[j][n] = 0.0f;
        }
    }

    const int64_t yw_row = (int64_t) p.nq*64;
    const int64_t ys_row = (int64_t) p.nq*2;
    const int nws = p.nq*13;

    for (int f = lane; f < nws; f += WARP_SIZE) {
        const int g  = f / 13;
        const int r  = f - g*13;
        const bool c1   = r < 8;
        const bool has5 = r < 12;
        const int y0 = c1 ? r : (has5 ? 32 + r : 60);  // 40 + (r - 8)
        const int ys = c1 ? 8 : (has5 ? 4 : 1);

        int   Y[ncols][5];
        float sl[ncols], sh[ncols];
        int   Sl[ncols], Sh[ncols];
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            const uint32_t * yb = p.yw + (ycol0 + j)*yw_row + g*64 + y0;
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                Y[j][t] = (int) yb[t*ys];
            }
            Y[j][4] = has5 ? (int) yb[4*ys] : 0;
            const float2 s = *((const float2 *) (p.ys + (ycol0 + j)*ys_row) + g);
            sl[j] = s.x;
            sh[j] = s.y;
            if (r == 0) {
                const int2 S = *((const int2 *) (p.yi + (ycol0 + j)*ys_row) + g);
                Sl[j] = S.x;
                Sh[j] = S.y;
            } else {
                Sl[j] = 0;
                Sh[j] = 0;
            }
        }

#pragma unroll
        for (int n = 0; n < R; ++n) {
            if (n < nr) {
                const char * wr = wrow + n*p.nb01;
                float al[ncols], ah[ncols];
#pragma unroll
                for (int j = 0; j < ncols; ++j) {
                    al[j] = 0.0f;
                    ah[j] = 0.0f;
                }
#pragma unroll
                for (int pl = 0; pl < K; ++pl) {
                    uint32_t w;
                    float d0, d1;
                    if constexpr (R1024) {
                        const int q = g & 3;
                        const uint32_t * Wb = (const uint32_t *) wr + ((int64_t) (g >> 2)*K + pl)*56;
                        w = Wb[has5 ? q*12 + r : 48 + q];
                        const float2 dd = __half22float2(*((const half2 *) (Wb + 52 + q)));
                        d0 = dd.x;
                        d1 = dd.y;
                    } else {
                        const char * b = wr + ((int64_t) g*K + pl)*54;
                        w  = tqvec_load_b2(b + 4*r);
                        d0 = __half2float(*((const half *) (b + 52)));
                        d1 = d0;
                    }
                    int D[5];
                    tqvec_decode(w, D);
#pragma unroll
                    for (int j = 0; j < ncols; ++j) {
                        const int ilo = ggml_cuda_dp4a(D[3], Y[j][3], ggml_cuda_dp4a(D[2], Y[j][2],
                                        ggml_cuda_dp4a(D[1], Y[j][1], ggml_cuda_dp4a(D[0], Y[j][0], 0)))) - Sl[j];
                        const int ihi = ggml_cuda_dp4a(D[4], Y[j][4], 0) - Sh[j];
                        const int lo  = c1 ? ilo : 0;
                        const int hi  = c1 ? ihi : ilo + ihi;
                        al[j] += d0*(float) lo;
                        ah[j] += d1*(float) hi;
                    }
                }
#pragma unroll
                for (int j = 0; j < ncols; ++j) {
                    temp[j][n] += sl[j]*al[j] + sh[j]*ah[j];
                }
            }
        }
    }

#pragma unroll
    for (int j = 0; j < ncols; ++j) {
#pragma unroll
        for (int n = 0; n < R; ++n) {
            const float v = warp_reduce_sum(temp[j][n]);
            if (lane == 0 && n < nr) {
                *(float *) ((char *) (drow + n) + j*p.nb1) = v;
            }
        }
    }
}

template <bool R1024, int K>
static void tqvec_launch_k(const tqvec_params & p, const int ncols, const int R, const int nwarps, const dim3 gyz,
                           cudaStream_t stream) {
    const dim3 block(WARP_SIZE, nwarps, 1);
    const int rows_per_block = (ncols == 1 ? R : 1)*nwarps;
    const dim3 grid((unsigned) ((p.nrows + rows_per_block - 1)/rows_per_block), gyz.y, gyz.z);
    switch (ncols) {
        case 1:
            switch (R) {
                case 1:  k_tqvec<R1024, K, 1, 1><<<grid, block, 0, stream>>>(p); break;
                case 2:  k_tqvec<R1024, K, 1, 2><<<grid, block, 0, stream>>>(p); break;
                case 4:  k_tqvec<R1024, K, 1, 4><<<grid, block, 0, stream>>>(p); break;
                default: GGML_ABORT("tq vec: unsupported rows per warp %d", R);
            }
            break;
        case 2: k_tqvec<R1024, K, 2, 1><<<grid, block, 0, stream>>>(p); break;
        case 3: k_tqvec<R1024, K, 3, 1><<<grid, block, 0, stream>>>(p); break;
        case 4: k_tqvec<R1024, K, 4, 1><<<grid, block, 0, stream>>>(p); break;
        case 5: k_tqvec<R1024, K, 5, 1><<<grid, block, 0, stream>>>(p); break;
        case 6: k_tqvec<R1024, K, 6, 1><<<grid, block, 0, stream>>>(p); break;
        case 7: k_tqvec<R1024, K, 7, 1><<<grid, block, 0, stream>>>(p); break;
        case 8: k_tqvec<R1024, K, 8, 1><<<grid, block, 0, stream>>>(p); break;
        default: GGML_ABORT("tq vec: unsupported ncols %d", ncols);
    }
}

template <bool R1024>
static void tqvec_launch(const tqvec_params & p, const int K, const int ncols, const int R, const int nwarps,
                         const dim3 gyz, cudaStream_t stream) {
    switch (K) {
        case 1: tqvec_launch_k<R1024, 1>(p, ncols, R, nwarps, gyz, stream); break;
        case 2: tqvec_launch_k<R1024, 2>(p, ncols, R, nwarps, gyz, stream); break;
        case 3: tqvec_launch_k<R1024, 3>(p, ncols, R, nwarps, gyz, stream); break;
        case 4: tqvec_launch_k<R1024, 4>(p, ncols, R, nwarps, gyz, stream); break;
        default: GGML_ABORT("tq vec: unsupported number of planes %d", K);
    }
}

void ggml_cuda_mul_mat_tq_vec(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                              const ggml_tensor * ids, ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS

    const ggml_type type = src0->type;
    const int  K      = ggml_cuda_tq_planes(type);
    const bool r1024  = ggml_cuda_tq_is_r1024(type);

    GGML_ASSERT(K > 0);
    GGML_ASSERT(src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ne10 == ne00);
    GGML_ASSERT(ne00 % (r1024 ? QK_TQ1_0R_1024 : QK_K) == 0);
    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));
    GGML_ASSERT(nb0 == sizeof(float));
    if (r1024) { // the kernel reads the weight as 32-bit words
        GGML_ASSERT(nb01 % 4 == 0 && nb02 % 4 == 0 && nb03 % 4 == 0 && ((uintptr_t) src0->data) % 4 == 0);
    } else {
        GGML_ASSERT(nb01 % 2 == 0 && nb02 % 2 == 0 && nb03 % 2 == 0 && ((uintptr_t) src0->data) % 2 == 0);
    }

    cudaStream_t stream = ctx.stream();

    const int     nq     = (int) (ne00 / QK_K);
    const int64_t ts1    = ggml_type_size(src1->type);
    const int64_t s11 = nb11/ts1, s12 = nb12/ts1, s13 = nb13/ts1;
    const int64_t nrows1 = ne11*ne12*ne13;

    const int64_t nw = (int64_t) nq*64, ns = (int64_t) nq*2;
    ggml_cuda_pool_alloc<uint32_t> ybuf(ctx.pool(), nrows1*(nw + 2*ns));
    uint32_t * yw = ybuf.get();
    float    * ys = (float *) (yw + nrows1*nw);
    int      * yi = (int   *) (yw + nrows1*(nw + ns));

    {
        const dim3 grid((unsigned) MIN((ne10 + 1023)/1024, (int64_t) 65535), (unsigned) MIN(nrows1, (int64_t) 65535), 1);
        if (src1->type == GGML_TYPE_F32) {
            if (r1024) {
                k_tqvec_q8<true,  float><<<grid, 256, 0, stream>>>((const float *) src1->data, yw, ys, yi, ne10, ne11, ne12, nrows1, s11, s12, s13);
            } else {
                k_tqvec_q8<false, float><<<grid, 256, 0, stream>>>((const float *) src1->data, yw, ys, yi, ne10, ne11, ne12, nrows1, s11, s12, s13);
            }
        } else {
            if (r1024) {
                k_tqvec_q8<true,  half><<<grid, 256, 0, stream>>>((const half *) src1->data, yw, ys, yi, ne10, ne11, ne12, nrows1, s11, s12, s13);
            } else {
                k_tqvec_q8<false, half><<<grid, 256, 0, stream>>>((const half *) src1->data, yw, ys, yi, ne10, ne11, ne12, nrows1, s11, s12, s13);
            }
        }
        CUDA_CHECK(cudaGetLastError());
    }

    tqvec_params p;
    p.w     = (const char *) src0->data;
    p.yw    = yw;
    p.ys    = ys;
    p.yi    = yi;
    p.dst   = (float *) dst->data;
    p.ids   = ids ? (const int32_t *) ids->data : nullptr;
    p.nq    = nq;
    p.nrows = ne01;
    p.nb01  = nb01; p.nb02 = nb02; p.nb03 = nb03;
    p.ne11  = ne11; p.ne12 = ne12;
    p.nb1   = nb1;  p.nb2  = nb2;  p.nb3  = nb3;
    p.r2    = ids ? 1 : ne12/ne02;
    p.r3    = ids ? 1 : ne13/ne03;
    p.ids_s0 = ids ? ids->nb[0]/ggml_type_size(ids->type) : 0;
    p.ids_s1 = ids ? ids->nb[1]/ggml_type_size(ids->type) : 0;

    // warps per block (GGML_CUDA_TQ_VEC_WARPS, default 4) and rows per warp for ncols == 1 (GGML_CUDA_TQ_VEC_ROWS
    // forces 1, 2 or 4): 4 rows when that still gives at least 2 blocks per SM, else 2 (measured on an RTX 4060:
    // 4 rows are faster for the large and MUL_MAT_ID shapes, 2 for a single 512-row matrix)
    static const int env_rows  = tqvec_env_int("GGML_CUDA_TQ_VEC_ROWS",  0);
    static const int env_warps = tqvec_env_int("GGML_CUDA_TQ_VEC_WARPS", 4);
    const int nwarps = env_warps >= 1 && env_warps <= 8 ? env_warps : 4;

    int ncols;
    dim3 gyz;
    if (ids) {
        GGML_ASSERT(ids->type == GGML_TYPE_I32);
        GGML_ASSERT(ne2 <= 65535);
        // dst: [ne0, n_expert_used, n_tokens]
        ncols = 1;
        gyz   = dim3(1, (unsigned) ne1, (unsigned) ne2);
    } else {
        GGML_ASSERT(ne12 % ne02 == 0 && ne13 % ne03 == 0);
        GGML_ASSERT(ne11 >= 1 && ne11 <= TQ1R_MMV_MAX_BATCH);
        ncols = (int) ne11;
        gyz   = dim3(1, (unsigned) ne12, (unsigned) ne13);
    }
    int R = env_rows == 1 || env_rows == 2 || env_rows == 4 ? env_rows : 0;
    if (R == 0) {
        const int nsm = ggml_cuda_info().devices[ctx.device].nsm;
        const int64_t blocks4 = (ne01 + 4*nwarps - 1)/(4*nwarps) * (int64_t) gyz.y * gyz.z;
        R = blocks4 >= 2*nsm ? 4 : 2;
    }
    if (r1024) {
        tqvec_launch<true >(p, K, ncols, R, nwarps, gyz, stream);
    } else {
        tqvec_launch<false>(p, K, ncols, R, nwarps, gyz, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}
