#pragma once

#include <torch/extension.h>

namespace fused_grid_sum {

void forward(torch::Tensor result,
             torch::Tensor image,
             torch::Tensor samples,
             torch::Tensor weights);

template <typename scalar_t>
__device__ scalar_t clamp(const scalar_t value,
                          const scalar_t min_value,
                          const scalar_t max_value) {
  return min(max(value, min_value), max_value);
}

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
    // Unravel the thread index.
    const int64_t i_b = (thread_index / result.stride(0)) % b;
    const int64_t i_hd = (thread_index / result.stride(1)) % hd;
    const int64_t i_s = (thread_index / result.stride(2)) % s;
    const int64_t i_c = (thread_index / result.stride(3)) % c;

    // Sum over the sample_summed dimension.
    const int64_t sample_summed = weights.size(3);
    const int64_t height = image.size(2);
    const int64_t width = image.size(3);
    scalar_t sum = 0;
    for (int32_t i = 0; i < sample_summed; i++) {
      // Get X and Y. To match the convention in torch.nn.functional.grid_sample,
      // we assume that samples are in range [-1, 1]. We convert to the full image range
      // here.
      const scalar_t x = (samples[i_b][i_s][i][0] * 0.5 + 0.5) * width;
      const scalar_t y = (samples[i_b][i_s][i][1] * 0.5 + 0.5) * height;

      // When grid sampling with align_corners=False, the outer 0.5 pixels on the border
      // of the image are not linearly interpolated. Thus, we clamp the sampling
      // coordinates. Also, interpolation effectively happens between pixel centers, so
      // we need to subtract 0.5 from the clamped values to get x/y values that can
      // easily be interpolated. The half variable is needed because of the scalar_t
      // template.
      const scalar_t half = 0.5;
      const scalar_t x_offset = clamp(x, half, width - half) - half;
      const scalar_t y_offset = clamp(y, half, height - half) - half;

      const int32_t col = static_cast<int32_t>(x_offset);
      const int32_t row = static_cast<int32_t>(y_offset);

      // Only add to the sum if the row and column are both valid. This is equivalent to
      // grid sample with zero padding.
      if (0 <= x && x <= width && 0 <= y && y <= height) {
        const scalar_t row_fraction = fmod(y_offset, 1);
        const scalar_t col_fraction = fmod(x_offset, 1);

        const scalar_t top_left = image[i_b][i_c][row][col];
        const scalar_t top_right = image[i_b][i_c][row][col + 1];
        const scalar_t bottom_left = image[i_b][i_c][row + 1][col];
        const scalar_t bottom_right = image[i_b][i_c][row + 1][col + 1];

        // Run horizontal linear interpolation.
        const scalar_t top = top_left * (1 - col_fraction) + col_fraction * top_right;
        const scalar_t bottom =
            bottom_left * (1 - col_fraction) + col_fraction * bottom_right;

        // Run vertical linear interpolation.
        const scalar_t interpolated = top * (1 - row_fraction) + bottom * row_fraction;

        // Multiply by the corresponding weight and sum.
        const scalar_t weight = weights[i_b][i_hd][i_s][i];
        sum += interpolated * weight;
      }
    }
    result[i_b][i_hd][i_s][i_c] = sum;
  }
}

}  // namespace fused_grid_sum
