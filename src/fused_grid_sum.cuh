#pragma once

#include <torch/extension.h>

namespace fused_grid_sum {

void forward(torch::Tensor result,
             torch::Tensor image,
             torch::Tensor samples,
             torch::Tensor weights);

template <typename scalar_t>
__global__ void forward_kernel(
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> result,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> image,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> samples,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits>
        weights) {
  // Create shorthands for dimension sizes.
  const int64_t b = result.size(0);  // batch
  const int64_t hd = result.size(1);  // head
  const int64_t s = result.size(2);  // sample_independent
  const int64_t c = result.size(3);  // channel
  const int64_t num_threads = b * hd * s * c;

  const int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_index < num_threads) {
    const int64_t i_b = (thread_index / result.stride(0)) % b;
    const int64_t i_hd = (thread_index / result.stride(1)) % hd;
    const int64_t i_s = (thread_index / result.stride(2)) % s;
    const int64_t i_c = (thread_index / result.stride(3)) % c;
    result[i_b][i_hd][i_s][i_c] = 123.0;
  }
  // if (sample_index < canvas.size(0)) {
  //   const glm::vec2 sample_xy = read_vec2(samples[sample_index]);

  //   // Iterate through the lines.
  //   glm::vec3 color{0.f};
  //   bool hit = false;
  //   for (int line_index = 0; line_index < starts.size(0); line_index++) {
  //     // Compute pixel-space start and end coordinates.
  //     const glm::vec2 start =
  //         world_to_pixel(read_vec2(starts[line_index]), bounds, image_shape);
  //     const glm::vec2 end =
  //         world_to_pixel(read_vec2(ends[line_index]), bounds, image_shape);

  //     // Define a vector between the start and end points.
  //     const glm::vec2 delta = end - start;
  //     const float delta_norm = glm::length(delta);
  //     const glm::vec2 u_delta = delta / delta_norm;

  //     // Define a vector between the sample and the start point.
  //     const glm::vec2 indicator = sample_xy - start;

  //     // Determine whether the sample is inside the line in the parallel direction.
  //     const int32_t cap = caps[line_index];
  //     const float width = widths[line_index];
  //     const float extra = (cap == CAP_SQUARE) ? 0.5f * width : 0.f;
  //     const float parallel = glm::dot(u_delta, indicator);
  //     const bool parallel_inside_line =
  //         (parallel <= delta_norm + extra) && (parallel > -extra);

  //     // Determine whether each sample is inside the line in the perpendicular
  //     // direction.
  //     const glm::vec2 perpendicular = indicator - parallel * u_delta;
  //     const bool perpendicular_inside_line = glm::length(perpendicular) < 0.5f *
  //     width;

  //     bool inside_line = parallel_inside_line && perpendicular_inside_line;

  //     // Compute round caps.
  //     if (cap == CAP_ROUND) {
  //       const bool near_start = glm::length(indicator) < 0.5f * width;
  //       inside_line |= near_start;
  //       const glm::vec2 end_indicator = sample_xy - end;
  //       const bool near_end = glm::length(end_indicator) < 0.5f * width;
  //       inside_line |= near_end;
  //     }

  //     // If inside the line, update the color.
  //     if (inside_line) {
  //       hit = true;
  //       color = read_vec3(colors[line_index]);
  //     }
  //   }

  //   // Write the resulting color.
  //   write_vec3(color, canvas[sample_index]);
  //   canvas[sample_index][3] = static_cast<float>(hit);
  // }
}

}  // namespace fused_grid_sum
