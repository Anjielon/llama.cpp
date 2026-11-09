# Ternary and multi-plane ternary types (TQ1_0R_1024, _X2/_X3/_X4, sparse expert planes)

This branch adds ternary weight types to ggml and kernels for Vulkan, CUDA, ROCm (HIP) and CPU.
It is the engine used to evaluate the Thor-T-35B-A3B XS / S / M models and the paper that describes them.

## Types

| type | block | content |
|---|---|---|
| `TQ1_0` | 256 values | one plane of trits (base-3 packed) + one fp16 scale (upstream type, llama.cpp PR #8151) |
| `TQ1_0_X2/_X3/_X4` | 256 values | 2/3/4 `TQ1_0` planes packed together; weight = sum of planes, each with its own scale |
| `TQ1_0R_1024` | 1024 values | one trit plane; the activation is rotated by a fixed Walsh–Hadamard transform with fixed signs per 1024-chunk before the dot product |
| `TQ1_0R_1024_X2/_X3/_X4` | 1024 values | 2/3/4 rotated planes packed together |

Mixture-of-experts tensors may carry a second, sparse plane: a separate tensor `ffn_*_exps2.weight` holding the
second plane only for some experts, plus an F32 slot map. Experts without a second plane have slot id `-1` and
contribute zero for that plane.

## Build

Vulkan (recommended on AMD and Intel GPUs):

    cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j

CUDA:

    cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j

ROCm (HIP):

    cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=<your gfx target> -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j

CPU only: `cmake -B build && cmake --build build -j`.

## Checks

    ./build/bin/test-backend-ops test -o MUL_MAT
    ./build/bin/test-backend-ops test -o MUL_MAT_ID

All ternary types must pass on the backend you built (they are compared against the CPU reference).

## Environment switches (all optional)

The paper's accuracy numbers used `GGML_VK_TQ1R_Q8=1` for S and M and the default (off) for XS; all other switches at their defaults.

Vulkan:
- `GGML_VK_TQ1R_Q8=1` — int8 activations in matrix-vector products for the rotated types (one scale per 256
  rotated values). Off by default. Faster on GPUs with integer dot products; not bit-identical to the float path.
- `GGML_VK_DISABLE_GRAPH_OPTIMIZE=1` — upstream switch; used in our likelihood benchmarks for run-to-run stability.
- `GGML_VK_TQ1_MMQ`, `GGML_VK_TQ1_MMQ_PLANES`, `GGML_VK_TQ1R_MM`, `GGML_VK_TQ1R_B16`, `GGML_VK_TQ1R_BK64_M`,
  `GGML_VK_TQ1R_ROWS`, `GGML_VK_TQ1_ROWS`, `GGML_VK_TQ1R_ID_ROWS`, `GGML_VK_TQ1R_WG_PER_CU`, `GGML_VK_TQ_WG`,
  `GGML_VK_TQ1_MMID_ADD`, `GGML_VK_TQ1_MMID_RES` — kernel selection and tuning knobs (tile sizes, rows per
  workgroup, batched-path selection). Leave unset unless you are tuning kernels.
- `GGML_VK_TQ1R_DEBUG=1` — prints which ternary kernel is chosen.

CUDA / ROCm:
- `GGML_CUDA_TQ_VEC=0` — use the previous MMVQ path instead of the ternary matrix-vector kernel.
- `GGML_CUDA_TQ_VEC_ROWS`, `GGML_CUDA_TQ_VEC_WARPS`, `GGML_CUDA_TQ_VEC_GS` — tuning of that kernel.
- `GGML_CUDA_TQ_MMQ=0` — disable the batched int8 (MMQ) path for ternary types.
- `GGML_CUDA_TQ1R_F32=1` — float activations for the rotated types (reference path).

CPU: no switches; AVX2 / AVX-512 kernels are selected at build time.

## Example

    ./build/bin/llama-server -m Thor-T-35B-A3B-M.gguf -ngl 999 -c 32768 --jinja

On GPUs with 8 GB, keep some expert layers on the CPU with `-ncmoe N` (e.g. M: 22–26, S: 18, XS: 14 on an
RTX 4060 Laptop 8 GB).
