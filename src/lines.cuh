#pragma once

#include <torch/extension.h>
#include <glm/geometric.hpp>
#include "conversions.cuh"
#include "glm_adapter.cuh"

namespace fused_grid_sum {

// Define line cap types.
constexpr int CAP_BUTT = 0;
constexpr int CAP_ROUND = 1;
constexpr int CAP_SQUARE = 2;

torch::Tensor render_lines(torch::Tensor canvas,
                           torch::Tensor samples,
                           torch::Tensor starts,
                           torch::Tensor ends,
                           torch::Tensor colors,
                           torch::Tensor widths,
                           torch::Tensor caps,
                           torch::Tensor bounds,
                           torch::Tensor image_shape);

template <typename scalar_t>
__global__ void render_lines_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> canvas,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> samples,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> starts,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> ends,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> colors,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> widths,
    const torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> caps,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> bounds,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits>
        image_shape) {
  const int sample_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample_index < canvas.size(0)) {
    const glm::vec2 sample_xy = read_vec2(samples[sample_index]);

    // Iterate through the lines.
    glm::vec3 color{0.f};
    bool hit = false;
    for (int line_index = 0; line_index < starts.size(0); line_index++) {
      // Compute pixel-space start and end coordinates.
      const glm::vec2 start =
          world_to_pixel(read_vec2(starts[line_index]), bounds, image_shape);
      const glm::vec2 end =
          world_to_pixel(read_vec2(ends[line_index]), bounds, image_shape);

      // Define a vector between the start and end points.
      const glm::vec2 delta = end - start;
      const float delta_norm = glm::length(delta);
      const glm::vec2 u_delta = delta / delta_norm;

      // Define a vector between the sample and the start point.
      const glm::vec2 indicator = sample_xy - start;

      // Determine whether the sample is inside the line in the parallel direction.
      const int32_t cap = caps[line_index];
      const float width = widths[line_index];
      const float extra = (cap == CAP_SQUARE) ? 0.5f * width : 0.f;
      const float parallel = glm::dot(u_delta, indicator);
      const bool parallel_inside_line =
          (parallel <= delta_norm + extra) && (parallel > -extra);

      // Determine whether each sample is inside the line in the perpendicular
      // direction.
      const glm::vec2 perpendicular = indicator - parallel * u_delta;
      const bool perpendicular_inside_line = glm::length(perpendicular) < 0.5f * width;

      bool inside_line = parallel_inside_line && perpendicular_inside_line;

      // Compute round caps.
      if (cap == CAP_ROUND) {
        const bool near_start = glm::length(indicator) < 0.5f * width;
        inside_line |= near_start;
        const glm::vec2 end_indicator = sample_xy - end;
        const bool near_end = glm::length(end_indicator) < 0.5f * width;
        inside_line |= near_end;
      }

      // If inside the line, update the color.
      if (inside_line) {
        hit = true;
        color = read_vec3(colors[line_index]);
      }
    }

    // Write the resulting color.
    write_vec3(color, canvas[sample_index]);
    canvas[sample_index][3] = static_cast<float>(hit);
  }
}

}  // namespace fused_grid_sum
