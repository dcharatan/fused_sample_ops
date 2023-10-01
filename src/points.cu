#include "points.cuh"

constexpr int BLOCK_SIZE = 512;

torch::Tensor fused_grid_sum::render_points(torch::Tensor canvas,
                                            torch::Tensor samples,
                                            torch::Tensor points,
                                            torch::Tensor colors,
                                            torch::Tensor outer_radii,
                                            torch::Tensor inner_radii,
                                            torch::Tensor bounds,
                                            torch::Tensor image_shape) {
  const int grid_size = (samples.size(0) + BLOCK_SIZE - 1) / BLOCK_SIZE;
  AT_DISPATCH_FLOATING_TYPES(
      samples.scalar_type(), "render_points_kernel", ([&] {
        render_points_kernel<scalar_t><<<grid_size, BLOCK_SIZE>>>(
            canvas.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            samples.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            points.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            colors.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            outer_radii.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            inner_radii.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            bounds.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            image_shape.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>());
      }));

  return samples;
}
