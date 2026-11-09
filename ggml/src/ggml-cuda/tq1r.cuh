#pragma once

// Ternary types on CUDA:
//   TQ1_0                     1.6875 bpw, upstream layout (one fp16 scale per 256 trits)
//   TQ1_0R_1024               1.75 bpw, four TQ1_0-layout quarters of 256 trits, one fp16 scale per 128,
//                             weights stored in a rotated basis (Hadamard-1024 + fixed sign vector S)
//   TQ1_0_X{2,3,4}            packed multi-plane variants: K base blocks of the same weights back to back,
//   TQ1_0R_1024_X{2,3,4}      the weight is the sum of the K planes
//
// For TQ1_0R_1024 a block of 1024 weights is w = S H (d.t), with H the normalized Sylvester-Hadamard
// transform (symmetric and orthogonal), S diagonal +-1 and d the per-128 scales. Hence
//     dot(w, x) = dot(d.t, H (S x)),
// so the activation is rotated once per 1024-block and the weights are only ever decoded "plain"
// (trit x d, exact in fp16). The K planes of a packed type share the same rotated activation.
//
// Reference arithmetic: ggml-quants.c (dequantize_row_*: these kernels reproduce it bit for bit) and the
// CPU vec_dot of these types (integer dot products against a per-256 int8 quantization of the rotated
// activation, the same granularity as the Q8_K activation of the CPU backend).

#include "common.cuh"
#include "convert.cuh"

// Number of planes (K) of a ternary type handled here, 0 for any other type.
static constexpr __host__ __device__ int ggml_cuda_tq_planes(const ggml_type type) {
    switch (type) {
        case GGML_TYPE_TQ1_0:
        case GGML_TYPE_TQ1_0R_1024:
            return 1;
        case GGML_TYPE_TQ1_0_X2:
        case GGML_TYPE_TQ1_0R_1024_X2:
            return 2;
        case GGML_TYPE_TQ1_0_X3:
        case GGML_TYPE_TQ1_0R_1024_X3:
            return 3;
        case GGML_TYPE_TQ1_0_X4:
        case GGML_TYPE_TQ1_0R_1024_X4:
            return 4;
        default:
            return 0;
    }
}

// TQ1_0R_1024 and its packed variants (rotated basis, own mat-vec and GEMM paths)
static constexpr __host__ __device__ bool ggml_cuda_tq_is_r1024(const ggml_type type) {
    return type == GGML_TYPE_TQ1_0R_1024    || type == GGML_TYPE_TQ1_0R_1024_X2 ||
           type == GGML_TYPE_TQ1_0R_1024_X3 || type == GGML_TYPE_TQ1_0R_1024_X4;
}

// TQ1_0 and its packed variants (MMVQ for small batches, plain-decode GEMM otherwise)
static constexpr __host__ __device__ bool ggml_cuda_tq_is_tq1_0(const ggml_type type) {
    return type == GGML_TYPE_TQ1_0    || type == GGML_TYPE_TQ1_0_X2 ||
           type == GGML_TYPE_TQ1_0_X3 || type == GGML_TYPE_TQ1_0_X4;
}

// Bytes of one plane (base block) of a ternary type: block_tq1_0 or block_tq1_0r_1024.
static constexpr __host__ __device__ size_t ggml_cuda_tq_plane_bytes(const ggml_type type) {
    return ggml_cuda_tq_is_r1024(type) ? sizeof(block_tq1_0r_1024) : sizeof(block_tq1_0);
}

// S, packed: bit (e & 31) of word (e >> 5) set <=> sign(e) = -1.
// Same 1024 values as ggml_tq1_0r_1024_sign[] in ggml-quants.c and tq1_0r_1024_sign_word() in the Vulkan
// shaders (numpy.random.default_rng(20260920).choice([-1.0, 1.0], size=1024)).
static __constant__ uint32_t tq1r_sign_words[32] = {
    0x4ee9d1aau, 0xffb3e643u, 0x8444b23bu, 0x18df7f70u, 0xfcd787ddu, 0x24065ad5u, 0xff6bf71eu, 0x9f5b8522u,
    0xc998dff5u, 0x6ee3d92du, 0x9c5cf507u, 0x15b1cef6u, 0x97a9a694u, 0x8eb6ac06u, 0x650fe151u, 0xca48d555u,
    0x2309815du, 0x01721db6u, 0x90b14a21u, 0x00ccd1f9u, 0x85554b92u, 0xa7ad0a28u, 0x737b15c6u, 0x9196684fu,
    0xbc3e4dd3u, 0xc4c90d73u, 0x980b0a9eu, 0x2171c76au, 0x4c4101b8u, 0x186554d3u, 0x7e2a4c8fu, 0x7d28e321u,
};

static __device__ __forceinline__ float tq1r_sign(const int e) {
    return ((tq1r_sign_words[e >> 5] >> (e & 31)) & 1u) ? -1.0f : 1.0f;
}

// Base-3 digit t (0..4) of a packed byte, in {0, 1, 2}. Same formula as the CPU reference: the 8-bit
// truncation of b*3^t is part of the algorithm.
static __device__ __forceinline__ int tq1_digit(const uint32_t b, const int t) {
    const uint32_t p = t == 0 ? 1u : t == 1 ? 3u : t == 2 ? 9u : t == 3 ? 27u : 81u;
    return (int) ((((b * p) & 255u) * 3u) >> 8);
}

// Trit {-1, 0, +1} of element e (0..255) of a block_tq1_0-layout group of 256 (qs: 48 bytes, qh: 4 bytes).
static __device__ __forceinline__ int tq1_trit(const uint8_t * qs, const uint8_t * qh, const int e) {
    uint32_t b;
    int t;
    if (e < 160) {
        b = qs[e & 31];                 t = e >> 5;
    } else if (e < 240) {
        const int e2 = e - 160;
        b = qs[32 + (e2 & 15)];         t = e2 >> 4;
    } else {
        const int e3 = e - 240;
        b = qh[e3 & 3];                 t = e3 >> 2;
    }
    return tq1_digit(b, t) - 1;
}

// In-place FWHT over 1024 floats in shared memory, 256 threads (tid < 256), without normalization. Same
// butterfly pairs, in the same stage order, as the CPU reference. ggml-cuda is compiled with -use_fast_math,
// so the float operations whose results must match the CPU bit for bit use explicit round-to-nearest intrinsics.
static __device__ __forceinline__ void tq1r_fwht1024(float * sh, const int tid) {
#pragma unroll
    for (int s = 0; s < 10; ++s) {
        const int len = 1 << s;
#pragma unroll
        for (int k = 0; k < 2; ++k) { // 512 butterflies per stage, 2 per thread
            const int pi = tid + 256*k;
            const int i = ((pi >> s) << (s + 1)) | (pi & (len - 1));
            const int j = i | len;
            const float u = sh[i];
            const float w = sh[j];
            sh[i] = __fadd_rn(u, w);
            sh[j] = __fsub_rn(u, w);
        }
        __syncthreads();
    }
}

// De-rotated dequantization of one packed block of K planes (x = its K consecutive base blocks), 256 threads.
// Bit-identical to dequantize_row_tq1_0r_1024[_xK]: per plane t*d, *1/32, butterflies, *S (the products by
// 2^-5 and +-1 are exact), then the planes are added in plane order.
template <int K, typename dst_t>
static __device__ __forceinline__ void tq1r_dequant_block(const block_tq1_0r_1024 * x, dst_t * y, float * sh) {
    const int tid = threadIdx.x;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int pl = 0; pl < K; ++pl) {
        const block_tq1_0r_1024 * xp = x + pl;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int   e = tid + 256*k;
            const int   q = e >> 8;
            const float d = __half2float(xp->d[e >> 7]);
            sh[e] = __fmul_rn(__fmul_rn((float) tq1_trit(xp->qs[q], xp->qh[q], e & 255), d), 0.03125f);
        }
        __syncthreads();
        tq1r_fwht1024(sh, tid);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int   e = tid + 256*k;
            const float v = __fmul_rn(sh[e], tq1r_sign(e));
            acc[k] = pl == 0 ? v : __fadd_rn(acc[k], v);
        }
        __syncthreads(); // sh is reused by the next plane or by the caller's next block
    }
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        y[tid + 256*k] = ggml_cuda_cast<dst_t>(acc[k]);
    }
}

// TQ1_0[_XK] super-block dequantization with the k-quant calling convention (64 threads, ib = index of the
// packed block). Bit-identical to dequantize_row_tq1_0[_xK]: trit*d per plane (exact), planes added in order.
template <int K, typename dst_t>
static __device__ __forceinline__ void dequantize_tq1_0_xk(const void * vx, const int64_t ib, dst_t * yy, const int tid) {
    const block_tq1_0 * x = (const block_tq1_0 *) vx + ib*K;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int e = tid + 64*k;
        float v = __fmul_rn((float) tq1_trit(x[0].qs, x[0].qh, e), __half2float(x[0].d));
#pragma unroll
        for (int pl = 1; pl < K; ++pl) {
            v = __fadd_rn(v, __fmul_rn((float) tq1_trit(x[pl].qs, x[pl].qh, e), __half2float(x[pl].d)));
        }
        yy[e] = ggml_cuda_cast<dst_t>(v);
    }
}

// GET_ROWS for TQ1_0R_1024[_XK] (de-rotated rows), same grid convention as k_get_rows_kq.
template <int K, typename dst_t>
static __global__ void __launch_bounds__(256) k_get_rows_tq1r(
        const void * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,
        const int64_t ne00, const int64_t ne11, const uint3 ne12_fdv,
        const size_t s1, const size_t s2, const size_t s3,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12) {
    __shared__ float sh[1024];
    const int64_t nsb = ne00/QK_TQ1_0R_1024;
    for (int64_t z = blockIdx.z; z < ne11*(int64_t)ne12_fdv.z; z += gridDim.z) {
        const int i10 = blockIdx.x;
        const uint2 dm = fast_div_modulo((uint32_t)z, ne12_fdv);
        const int i11 = dm.x;
        const int i12 = dm.y;

        const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

        dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
        const block_tq1_0r_1024 * src0_row = (const block_tq1_0r_1024 *) ((const char *) src0 + i01*nb01 + i11*nb02 + i12*nb03);

        for (int64_t ib = blockIdx.y; ib < nsb; ib += gridDim.y) {
            tq1r_dequant_block<K>(src0_row + ib*K, dst_row + ib*QK_TQ1_0R_1024, sh);
        }
    }
}

template <int K, typename dst_t>
static void get_rows_cuda_tq1r(
        const void * src0_d, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_TQ1_0R_1024 == 0);
    GGML_ASSERT(ne12 > 0);
    const int64_t nsb = ne00/QK_TQ1_0R_1024;
    const dim3 block_dims(256, 1, 1);
    const dim3 block_nums(ne10, MIN(nsb, UINT16_MAX), MIN(ne11*ne12, UINT16_MAX));
    const uint3 ne12_fdv = init_fastdiv_values(ne12);
    k_get_rows_tq1r<K, dst_t><<<block_nums, block_dims, 0, stream>>>(
        src0_d, src1_d, dst_d, ne00, ne11, ne12_fdv,
        nb1/sizeof(dst_t), nb2/sizeof(dst_t), nb3/sizeof(dst_t),
        nb01, nb02, nb03,
        nb10/sizeof(int32_t), nb11/sizeof(int32_t), nb12/sizeof(int32_t));
}

// Contiguous dequantization of TQ1_0R_1024[_XK] (convert.cu entry points), nullptr for other types.
to_fp32_cuda_t ggml_cuda_tq1r_get_to_fp32(ggml_type type);
to_fp16_cuda_t ggml_cuda_tq1r_get_to_fp16(ggml_type type);

// MUL_MAT / MUL_MAT_ID for TQ1_0R_1024[_XK].
//  - ids == nullptr: MUL_MAT. ne11 <= TQ1R_MMV_MAX_BATCH and !force_batch -> int8 mat-vec,
//    otherwise rotated-activation GEMM (fp16 operands, fp32 accumulation; GGML_CUDA_TQ1R_F32=1: fp32 SGEMM).
//  - ids != nullptr: MUL_MAT_ID, only for n_tokens <= TQ1R_MMID_MAX_BATCH (int8 mat-vec per (token, slot)).
void ggml_cuda_mul_mat_tq1r(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                            const ggml_tensor * ids, ggml_tensor * dst, bool force_batch);

// Rotated activation for the MMQ path of TQ1_0R_1024[_XK]: y = (1/32) H_1024 (S x) for every 1024-chunk of every
// row of the strided [ne10, ne11, ne12, ne13] x (element strides s11, s12, s13); y is contiguous.
void ggml_cuda_tq1r_rotate_f32(const float * x, float * y, int64_t ne10, int64_t ne11, int64_t ne12, int64_t ne13,
                               int64_t s11, int64_t s12, int64_t s13, cudaStream_t stream);

// Batched GEMM for every ternary type above: plain (exact) fp16 decode of each plane, rotated activation for the
// TQ1_0R_1024 family, one fp32-accumulated GEMM per plane. Used for TQ1_0[_XK] batches (instead of the generic
// cuBLAS path, which would round the sum of the planes to fp16) and by ggml_cuda_mul_mat_tq1r for large batches.
void ggml_cuda_mul_mat_tq_batch(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// Mat-vec for every ternary type (tqvec.cu): MUL_MAT with ne11 <= TQ1R_MMV_MAX_BATCH, MUL_MAT_ID with
// n_tokens <= TQ1R_MMID_MAX_BATCH (one (token, slot) per launch column; ids < 0 give zero rows).
// Default path; GGML_CUDA_TQ_VEC=0 selects the previous kernels (MMVQ for the TQ1_0 family, the byte-order
// int8 mat-vec of tq1r.cu for the TQ1_0R_1024 family).
bool ggml_cuda_tq_vec_enabled();
void ggml_cuda_mul_mat_tq_vec(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                              const ggml_tensor * ids, ggml_tensor * dst);

// MUL_MAT_ID: max number of tokens served by the int8 mat-vec kernel.
#define TQ1R_MMID_MAX_BATCH 8
// MUL_MAT: max ne11 served by the int8 mat-vec kernel (= MMVQ_MAX_BATCH_SIZE).
#define TQ1R_MMV_MAX_BATCH 8
