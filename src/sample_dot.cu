#include "common.cuh"
#include "sample_dot.cuh"

constexpr int BLOCK_SIZE = 256;

void fused_grid_ops::sample_dot_forward(torch::Tensor images,
                                        torch::Tensor samples,
                                        torch::Tensor queries,
                                        torch::Tensor depths,
                                        torch::Tensor outputs) {
  // We assume that 32-bit indexing can be used and that only float32 and float64 are
  // supported.
  int B = images.size(0);
  int HD = queries.size(1);
  int Q = samples.size(1);
  int D = samples.size(2);
  int num_threads = B * HD * Q * D;
  if (num_threads > 0) {
    AT_DISPATCH_FLOATING_TYPES(
        images.scalar_type(), "sample_dot_forward", ([&] {
          sample_dot_forward_kernel<scalar_t>
              <<<get_blocks(num_threads, BLOCK_SIZE), BLOCK_SIZE>>>(
                  num_threads,
                  images.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                  samples.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                  queries.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                  depths.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                  outputs.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>());
        }));
  }
}

void fused_grid_ops::sample_dot_backward(torch::Tensor output_gradients,
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
        images.scalar_type(), "sample_dot_backward", ([&] {
          sample_dot_backward_kernel<scalar_t>
              <<<get_blocks(num_threads, BLOCK_SIZE), BLOCK_SIZE>>>(
                  num_threads,
                  output_gradients
                      .packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                  images.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                  samples.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                  queries.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                  depths.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                  image_gradients
                      .packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                  sample_gradients
                      .packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                  query_gradients
                      .packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                  depth_gradients
                      .packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>());
        }));
  }
}