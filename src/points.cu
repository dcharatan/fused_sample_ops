#include "points.cuh"

constexpr int BLOCK_SIZE = 512;

torch::Tensor torchdraw_cuda::render_points(torch::Tensor samples,
                            torch::Tensor points,
                            torch::Tensor colors,
                            torch::Tensor outer_radii,
                            torch::Tensor inner_radii,
                            torch::Tensor bounds) {
  const int grid_size = (samples.size(0) + BLOCK_SIZE - 1) / BLOCK_SIZE;
  AT_DISPATCH_FLOATING_TYPES(
      samples.type(), "render_points_kernel", ([&] {
        render_points_kernel<scalar_t><<<grid_size, BLOCK_SIZE>>>(
            samples.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>());
      }));

  return samples;
}