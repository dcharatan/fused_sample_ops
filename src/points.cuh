#pragma once

#include <torch/extension.h>
#include <glm/geometric.hpp>
#include "conversions.cuh"
#include "glm_adapter.cuh"

namespace torchdraw_cuda {

torch::Tensor render_points(torch::Tensor canvas,
                            torch::Tensor samples,
                            torch::Tensor points,
                            torch::Tensor colors,
                            torch::Tensor outer_radii,
                            torch::Tensor inner_radii,
                            torch::Tensor bounds,
                            torch::Tensor image_shape);

template <typename scalar_t>
__global__ void render_points_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> canvas,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> samples,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> points,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> colors,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits>
        outer_radii,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits>
        inner_radii,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> bounds,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits>
        image_shape) {
  const int sample_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample_index < canvas.size(0)) {
    const glm::vec2 sample_xy = read_vec2(samples[sample_index]);

    // Iterate through the points.
    glm::vec3 color{0.f};
    bool hit;
    for (int point_index = 0; point_index < points.size(0); point_index++) {
      // Compute the pixel-space distance between the point and the sample.
      const glm::vec2 point_world_xy = read_vec2(points[point_index]);
      const glm::vec2 point_pixel_xy =
          world_to_pixel(point_world_xy, bounds, image_shape);
      const float distance = glm::length(sample_xy - point_pixel_xy);

      // If the distance indicates a hit, update the color.
      if (distance <= outer_radii[sample_index] &&
          distance >= inner_radii[sample_index]) {
        hit = true;
        color = read_vec3(colors[sample_index]);
      }
    }

    // Write the resulting color.
    write_vec3(color, canvas[sample_index]);
    canvas[sample_index][3] = static_cast<float>(hit);
  }
}

}  // namespace torchdraw_cuda
