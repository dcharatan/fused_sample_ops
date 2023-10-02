#pragma once

#include <torch/extension.h>

namespace fused_grid_sum {

void forward(torch::Tensor results,
             torch::Tensor images,
             torch::Tensor samples,
             torch::Tensor weights);

// We don't compute gradients for the sample positions.
void backward(torch::Tensor result_gradients,
              torch::Tensor images,
              torch::Tensor image_gradients,
              torch::Tensor samples,
              torch::Tensor weights,
              torch::Tensor weight_gradients);

template <typename scalar_t>
__device__ scalar_t clamp(const scalar_t value,
                          const scalar_t min_value,
                          const scalar_t max_value) {
  return min(max(value, min_value), max_value);
}

template <typename scalar_t>
__global__ void forward_kernel(
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> results,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> images,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> samples,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits>
        weights) {
  // Create shorthands for dimension sizes.
  const int32_t b = results.size(0);  // batch
  const int32_t hd = results.size(1);  // head
  const int32_t s = results.size(2);  // sample_independent
  const int32_t c = results.size(3);  // channel
  const int32_t num_threads = b * hd * s * c;

  const int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_index < num_threads) {
    // Unravel the thread index.
    const int32_t i_b = (thread_index / results.stride(0)) % b;
    const int32_t i_hd = (thread_index / results.stride(1)) % hd;
    const int32_t i_s = (thread_index / results.stride(2)) % s;
    const int32_t i_c = (thread_index / results.stride(3)) % c;

    // Sum over the sample_summed dimension.
    const int32_t sample_summed = weights.size(3);
    const int32_t height = images.size(2);
    const int32_t width = images.size(3);
    scalar_t sum = 0;
    for (int32_t i = 0; i < sample_summed; i++) {
      // Get X and Y. To match the convention in torch.nn.functional.grid_sample,
      // we assume that samples are in range [-1, 1].
      const scalar_t x = (samples[i_b][i_s][i][0] * 0.5 + 0.5) * width + 0.5;
      const scalar_t y = (samples[i_b][i_s][i][1] * 0.5 + 0.5) * height + 0.5;

      // Only add to the sum if both the width and height are in bounds.
      if (0 <= x && x < width + 1 && 0 <= y && y < height + 1) {
        const int32_t col = static_cast<int32_t>(x) - 1;
        const int32_t row = static_cast<int32_t>(y) - 1;
        const scalar_t row_fraction = fmod(y, 1);
        const scalar_t col_fraction = fmod(x, 1);

        const int32_t row_n = row + 1;
        const int32_t col_n = col + 1;

        const scalar_t top_left =
            (row == -1 || col == -1) ? 0 : images[i_b][i_c][row][col];
        const scalar_t top_right =
            (row == -1 || col_n == width) ? 0 : images[i_b][i_c][row][col_n];
        const scalar_t bottom_left =
            (row_n == height || col == -1) ? 0 : images[i_b][i_c][row_n][col];
        const scalar_t bottom_right =
            (row_n == height || col_n == width) ? 0 : images[i_b][i_c][row_n][col_n];

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
    results[i_b][i_hd][i_s][i_c] = sum;
  }
}

template <typename scalar_t>
__global__ void backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits>
        result_gradients,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> images,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits>
        image_gradients,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> samples,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> weights,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits>
        weight_gradients) {
  // Create shorthands for dimension sizes.
  const int32_t b = result_gradients.size(0);  // batch
  const int32_t hd = result_gradients.size(1);  // head
  const int32_t s = result_gradients.size(2);  // sample_independent
  const int32_t c = result_gradients.size(3);  // channel
  const int32_t num_threads = b * hd * s * c;

  const int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_index < num_threads) {
    // Unravel the thread index.
    const int32_t i_b = (thread_index / result_gradients.stride(0)) % b;
    const int32_t i_hd = (thread_index / result_gradients.stride(1)) % hd;
    const int32_t i_s = (thread_index / result_gradients.stride(2)) % s;
    const int32_t i_c = (thread_index / result_gradients.stride(3)) % c;

    // Repeat the forward pass in the backward pass. This allows us to avoid storing the
    // grid-sampled tensor at any point during the forward and backward passes.
    const int32_t sample_summed = weights.size(3);
    const int32_t height = images.size(2);
    const int32_t width = images.size(3);
    scalar_t sum = 0;
    for (int32_t i = 0; i < sample_summed; i++) {
      // Get X and Y. To match the convention in torch.nn.functional.grid_sample,
      // we assume that samples are in range [-1, 1].
      const scalar_t x = (samples[i_b][i_s][i][0] * 0.5 + 0.5) * width + 0.5;
      const scalar_t y = (samples[i_b][i_s][i][1] * 0.5 + 0.5) * height + 0.5;

      // Only add to the sum if both the width and height are in bounds.
      if (0 <= x && x < width + 1 && 0 <= y && y < height + 1) {
        const int32_t col = static_cast<int32_t>(x) - 1;
        const int32_t row = static_cast<int32_t>(y) - 1;
        const scalar_t row_fraction = fmod(y, 1);
        const scalar_t col_fraction = fmod(x, 1);

        const int32_t row_n = row + 1;
        const int32_t col_n = col + 1;

        const scalar_t result_gradient = result_gradients[i_b][i_hd][i_s][i_c];
        const scalar_t weight = weights[i_b][i_hd][i_s][i];
        const scalar_t weight_with_grad = weight * result_gradient;

        // Compute gradients for the top left sample.
        if (row != -1 && col != -1) {
          const scalar_t top_left_grad =
              weight_with_grad * (1 - col_fraction) * (1 - row_fraction);
          atomicAdd(&image_gradients[i_b][i_c][row][col], top_left_grad);
        }

        // Compute gradients for the top right sample.
        if (row != -1 && col_n != width) {
          const scalar_t top_right_grad =
              weight_with_grad * col_fraction * (1 - row_fraction);
          atomicAdd(&image_gradients[i_b][i_c][row][col_n], top_right_grad);
        }

        // Compute gradients for the bottom left sample.
        if (row_n != height && col != -1) {
          const scalar_t bottom_left_grad =
              weight_with_grad * (1 - col_fraction) * row_fraction;
          atomicAdd(&image_gradients[i_b][i_c][row_n][col], bottom_left_grad);
        }

        // Compute gradients for the bottom right sample.
        if (row_n != height && col_n != width) {
          const scalar_t bottom_right_grad =
              weight_with_grad * col_fraction * row_fraction;
          atomicAdd(&image_gradients[i_b][i_c][row_n][col_n], bottom_right_grad);
        }

        // Recompute the value of the interpolation so that we can update the weight
        // gradient.
        const scalar_t top_left =
            (row == -1 || col == -1) ? 0 : images[i_b][i_c][row][col];
        const scalar_t top_right =
            (row == -1 || col_n == width) ? 0 : images[i_b][i_c][row][col_n];
        const scalar_t bottom_left =
            (row_n == height || col == -1) ? 0 : images[i_b][i_c][row_n][col];
        const scalar_t bottom_right =
            (row_n == height || col_n == width) ? 0 : images[i_b][i_c][row_n][col_n];

        // Run horizontal linear interpolation.
        const scalar_t top = top_left * (1 - col_fraction) + col_fraction * top_right;
        const scalar_t bottom =
            bottom_left * (1 - col_fraction) + col_fraction * bottom_right;

        // Run vertical linear interpolation.
        const scalar_t interpolated = top * (1 - row_fraction) + bottom * row_fraction;

        // Compute gradients for the weights.
        atomicAdd(&weight_gradients[i_b][i_hd][i_s][i], interpolated * result_gradient);
      }
    }
  }
}

}  // namespace fused_grid_sum
