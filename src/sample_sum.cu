#include "common.cuh"
#include "sample_sum.cuh"

void fused_sample_ops::sample_sum_forward(torch::Tensor images,
                                          torch::Tensor samples,
                                          torch::Tensor weights,
                                          torch::Tensor outputs) {
  // We assume that 32-bit indexing can be used and that only float32 and float64 are
  // supported.
  int B = weights.size(0);
  int HD = weights.size(1);
  int Q = weights.size(2);
  int D = weights.size(3);
  int num_threads = B * Q * D;
  if (num_threads > 0) {
    AT_DISPATCH_FLOATING_TYPES(
        images.scalar_type(), "sample_sum_forward", ([&] {
          sample_sum_forward_kernel<scalar_t>
              <<<get_blocks(num_threads, BLOCK_SIZE), BLOCK_SIZE,
                 BLOCK_SIZE * HD * sizeof(scalar_t)>>>(
                  num_threads,
                  images.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                  samples.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                  weights.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                  outputs.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>());
        }));
  }
}

void fused_sample_ops::sample_sum_backward(torch::Tensor output_gradients,
                                           torch::Tensor images,
                                           torch::Tensor samples,
                                           torch::Tensor weights,
                                           torch::Tensor image_gradients,
                                           torch::Tensor sample_gradients,
                                           torch::Tensor weight_gradients) {
  // We assume that 32-bit indexing can be used and that only float32 and float64 are
  // supported. Note that we parallelize over different dimensions than in the forward
  // pass, since the backward pass has no dependency between depths.
  int B = weights.size(0);
  int Q = weights.size(2);
  int D = weights.size(3);
  int num_threads = B * Q * D;
  if (num_threads > 0) {
    AT_DISPATCH_FLOATING_TYPES(
        images.scalar_type(), "sample_sum_backward", ([&] {
          sample_sum_backward_kernel<scalar_t>
              <<<get_blocks(num_threads, BLOCK_SIZE), BLOCK_SIZE>>>(
                  num_threads,
                  output_gradients
                      .packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                  images.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                  samples.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                  weights.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                  image_gradients
                      .packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                  sample_gradients
                      .packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                  weight_gradients
                      .packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>());
        }));
  }
}