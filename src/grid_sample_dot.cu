#include "common.cuh"
#include "grid_sample_dot.cuh"

constexpr int BLOCK_SIZE = 256;

void fused_grid_sum::grid_sample_dot_forward(torch::Tensor images,
                                             torch::Tensor samples,
                                             torch::Tensor outputs) {
  // We assume that 32-bit indexing can be used and that only float32 and float64 are
  // supported.
  int b = images.size(0);
  int h = samples.size(1);
  int w = samples.size(2);
  int num_threads = b * h * w;
  if (num_threads > 0) {
    AT_DISPATCH_FLOATING_TYPES(
        images.scalar_type(), "grid_sample_dot_forward", ([&] {
          grid_sample_dot_forward_kernel<scalar_t>
              <<<get_blocks(num_threads, BLOCK_SIZE), BLOCK_SIZE>>>(
                  num_threads,
                  images.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                  samples.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                  outputs.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>());
        }));
  }
}