#pragma once

// These functions are adapted or taken directly from the PyTorch functions here:
// https://github.com/pytorch/pytorch/blob/7e6cf04a843a645c662ffb2eb4334ed84da97f01/aten/src/ATen/cuda/detail/KernelUtils.h

namespace fused_grid_sum {

inline int get_blocks(const int64_t N, const int64_t max_threads_per_block) {
  auto block_num = (N - 1) / max_threads_per_block + 1;
  return static_cast<int>(block_num);
}

// CUDA: grid stride looping
// int64_t _i_n_d_e_x specifically prevents overflow in the loop increment.
// If input.numel() < INT_MAX, _i_n_d_e_x < INT_MAX, except after the final
// iteration of the loop where _i_n_d_e_x += blockDim.x * gridDim.x can be
// greater than INT_MAX.  But in that case _i_n_d_e_x >= n, so there are no
// further iterations and the overflowed value in i=_i_n_d_e_x is not used.
#define CUDA_KERNEL_LOOP_TYPE(i, n, index_type)               \
  int64_t _i_n_d_e_x = blockIdx.x * blockDim.x + threadIdx.x; \
  for (index_type i = _i_n_d_e_x; _i_n_d_e_x < (n);           \
       _i_n_d_e_x += blockDim.x * gridDim.x, i = _i_n_d_e_x)

// This is significantly simpler than the original PyTorch function because we assume
// align_corners is false and the padding mode is zeros.
template <typename scalar_t>
__device__ __forceinline__ scalar_t grid_sampler_compute_source_index(scalar_t coord,
                                                                      int64_t size) {
  return ((coord + 1) * size - 1) / 2;
}

__device__ __forceinline__ bool within_bounds_2d(int64_t h,
                                                 int64_t w,
                                                 int64_t H,
                                                 int64_t W) {
  return h >= 0 && h < H && w >= 0 && w < W;
}

}  // namespace fused_grid_sum