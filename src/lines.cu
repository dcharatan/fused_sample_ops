#include "lines.cuh"

constexpr int BLOCK_SIZE = 512;

torch::Tensor torchdraw_cuda::render_lines(torch::Tensor canvas,
                                           torch::Tensor samples,
                                           torch::Tensor starts,
                                           torch::Tensor ends,
                                           torch::Tensor colors,
                                           torch::Tensor widths,
                                           torch::Tensor caps,
                                           torch::Tensor bounds,
                                           torch::Tensor image_shape) {
  const int grid_size = (samples.size(0) + BLOCK_SIZE - 1) / BLOCK_SIZE;
  AT_DISPATCH_FLOATING_TYPES(
      samples.scalar_type(), "render_lines_kernel", ([&] {
        render_lines_kernel<scalar_t><<<grid_size, BLOCK_SIZE>>>(
            canvas.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            samples.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            starts.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            ends.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            colors.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            widths.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            caps.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
            bounds.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            image_shape.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>());
      }));

  return samples;
}
