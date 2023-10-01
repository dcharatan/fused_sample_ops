#include "fused_grid_sum.cuh"

constexpr int BLOCK_SIZE = 512;

torch::Tensor fused_grid_sum::forward(torch::Tensor result,
                                      torch::Tensor image,
                                      torch::Tensor samples,
                                      torch::Tensor weights) {
  // Compute the number of threads. For now, just parallelize over independent samples.
  const int64_t batch = weights.size(0);
  const int64_t sample_independent = weights.size(1);
  const int64_t head = weights.size(2);
  const int64_t num_threads = batch * sample_independent * head;

  const int grid_size = (num_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;
  AT_DISPATCH_FLOATING_TYPES(
      samples.scalar_type(), "forward_kernel", ([&] {
        forward_kernel<scalar_t><<<grid_size, BLOCK_SIZE>>>(
            result.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            image.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            samples.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            weights.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>());
      }));

  return samples;
}
