#include <iostream>
#include "fused_grid_sum.cuh"

constexpr int BLOCK_SIZE = 512;

void fused_grid_sum::forward(torch::Tensor result,
                             torch::Tensor image,
                             torch::Tensor samples,
                             torch::Tensor weights) {
  // Compute the number of threads. For now, just parallelize over independent samples.
  const int64_t b = result.size(0);  // batch
  const int64_t hd = result.size(1);  // head
  const int64_t s = result.size(2);  // sample_independent
  const int64_t c = result.size(3);  // channel
  const int64_t num_threads = b * hd * s * c;

  const int grid_size = (num_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;
  AT_DISPATCH_FLOATING_TYPES(
      samples.scalar_type(), "forward_kernel", ([&] {
        forward_kernel<scalar_t><<<grid_size, BLOCK_SIZE>>>(
            result.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            image.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            samples.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            weights.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>());
      }));
}
