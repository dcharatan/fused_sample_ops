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
      // we assume that each image axis is in range [-1, 1]. We convert to range [0, 1].
      const scalar_t x = samples[i_b][i_s][i][0] * 0.5 + 0.5;
      const scalar_t y = samples[i_b][i_s][i][1] * 0.5 + 0.5;
      const int32_t col = static_cast<int32_t>(x * width);
      const int32_t row = static_cast<int32_t>(y * height);

      // Only add to the sum if the row and column are both valid. This is equivalent to
      // grid sample with zero padding.
      if (0 <= row && row < height && 0 <= col && col < width) {
        const scalar_t row_fraction = fmod(y * height, 1.f);
        const scalar_t col_fraction = fmod(x * width, 1.f);

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
