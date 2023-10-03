#include "common.cuh"
#include "grid_sample_dot.cuh"

constexpr int BLOCK_SIZE = 256;

void fused_grid_sum::grid_sample_dot_forward(torch::Tensor images,
                                             torch::Tensor samples,
                                             torch::Tensor queries,
                                             torch::Tensor depths,
                                             torch::Tensor outputs) {
  // We assume that 32-bit indexing can be used and that only float32 and float64 are
  // supported.
  int B = images.size(0);
  int Q = samples.size(1);
  int D = samples.size(2);
  int num_threads = B * Q * D;
  if (num_threads > 0) {
    AT_DISPATCH_FLOATING_TYPES(
        images.scalar_type(), "grid_sample_dot_forward", ([&] {
          grid_sample_dot_forward_kernel<scalar_t>
              <<<get_blocks(num_threads, BLOCK_SIZE), BLOCK_SIZE>>>(
                  num_threads,
                  images.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                  samples.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                  queries.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                  depths.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                  outputs.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>());
        }));
  }
}

void fused_grid_sum::grid_sample_dot_backward(torch::Tensor result_gradients,
                                              torch::Tensor images,
                                              torch::Tensor samples,
                                              torch::Tensor queries,
                                              torch::Tensor depths,
                                              torch::Tensor image_gradients,
                                              torch::Tensor sample_gradients,
                                              torch::Tensor query_gradients,
                                              torch::Tensor depth_gradients) {
  // We assume that 32-bit indexing can be used and that only float32 and float64 are
  // supported. We also assume that all tensors (except samples) need a gradient.
  int B = images.size(0);
  int Q = samples.size(1);
  int D = samples.size(2);
  int num_threads = B * Q * D;
  if (num_threads > 0) {
    AT_DISPATCH_FLOATING_TYPES(
        images.scalar_type(), "grid_sample_dot_backward", ([&] {
          grid_sample_dot_backward_kernel<scalar_t>
              <<<get_blocks(num_threads, BLOCK_SIZE), BLOCK_SIZE>>>(
                  num_threads,
                  result_gradients
                      .packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                  images.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                  samples.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                  queries.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                  depths.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                  image_gradients
                      .packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                  sample_gradients
                      .packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                  query_gradients
                      .packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                  depth_gradients
                      .packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>());
        }));
  }
}