#include <iostream>
#include "fused_grid_sum.cuh"

constexpr int BLOCK_SIZE = 512;

void fused_grid_sum::forward(torch::Tensor results,
                             torch::Tensor images,
                             torch::Tensor samples,
                             torch::Tensor weights) {
  // Compute the number of threads. For now, just parallelize over independent samples.
  const int64_t b = results.size(0);  // batch
  const int64_t hd = results.size(1);  // head
  const int64_t s = results.size(2);  // sample_independent
  const int64_t c = results.size(3);  // channel
  const int64_t num_threads = b * hd * s * c;

  const int grid_size = (num_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;
  AT_DISPATCH_FLOATING_TYPES(
      samples.scalar_type(), "forward_kernel", ([&] {
        forward_kernel<scalar_t><<<grid_size, BLOCK_SIZE>>>(
            results.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            images.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            samples.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            weights.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>());
      }));
}

void fused_grid_sum::backward(torch::Tensor results,
                        torch::Tensor result_gradients,
                             torch::Tensor images,
                             torch::Tensor image_gradients,
                             torch::Tensor samples,
                             torch::Tensor weights,
                             torch::Tensor weight_gradients) {
  // Compute the number of threads. For now, just parallelize over independent samples.
  const int64_t b = results.size(0);  // batch
  const int64_t hd = results.size(1);  // head
  const int64_t s = results.size(2);  // sample_independent
  const int64_t c = results.size(3);  // channel
  const int64_t num_threads = b * hd * s * c;

  const int grid_size = (num_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;
  AT_DISPATCH_FLOATING_TYPES(
      samples.scalar_type(), "forward_kernel", ([&] {
        forward_kernel<scalar_t><<<grid_size, BLOCK_SIZE>>>(
            results.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            images.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            samples.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            weights.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>());
      }));
}
