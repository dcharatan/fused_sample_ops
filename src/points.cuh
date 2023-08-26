#pragma once

#include <torch/extension.h>

namespace torchdraw_cuda {

torch::Tensor render_points(torch::Tensor samples,
                            torch::Tensor points,
                            torch::Tensor colors,
                            torch::Tensor outer_radii,
                            torch::Tensor inner_radii,
                            torch::Tensor bounds);

template <typename scalar_t>
__global__ void render_points_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>
        samples) {
  const int sample_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample_index < samples.size(0)) {
    samples[sample_index][0] = 1.f;
    samples[sample_index][1] = 0.5f;
    samples[sample_index][2] = 0.25f;
  }
}

}  // namespace torchdraw_cuda
