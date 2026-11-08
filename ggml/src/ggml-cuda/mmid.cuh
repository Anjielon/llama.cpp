#pragma once

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds,
        int n_experts, int n_tokens, int n_expert_used, int nchannels_y, int si1, int sis1, bool write_inverse, cudaStream_t stream);

// Like ggml_cuda_launch_mm_ids_helper, but one sorted row per (token, slot): an expert repeated within a token gets a
// row per slot, a slot with a negative id gets none (expert_bounds counts only the valid slots). Forward map only.
void ggml_cuda_launch_mm_ids_helper_slots(
        const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds,
        int n_experts, int n_tokens, int n_expert_used, int nchannels_y, int si1, int sis1, cudaStream_t stream);
