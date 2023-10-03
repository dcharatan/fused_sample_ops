#pragma once

#include <torch/extension.h>

#include "common.cuh"

namespace fused_grid_sum {

void sample_sum_forward(torch::Tensor images,
                        torch::Tensor samples,
                        torch::Tensor weights,
                        torch::Tensor outputs);

void sample_sum_backward(torch::Tensor output_gradients,
                         torch::Tensor images,
                         torch::Tensor samples,
                         torch::Tensor weights,
                         torch::Tensor image_gradients,
                         torch::Tensor sample_gradients,
                         torch::Tensor weight_gradients);

template <typename scalar_t, typename index_t>
__launch_bounds__(256) __global__ void sample_sum_forward_kernel(
    const index_t num_threads,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> images,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> samples,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> weights,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> outputs) {
  // Extract dimensions.
  const index_t B = images.size(0);
  const index_t C = images.size(1);
  const index_t H = images.size(2);
  const index_t W = images.size(3);
  const index_t HD = weights.size(1);
  const index_t Q = weights.size(2);
  const index_t D = weights.size(3);

  CUDA_KERNEL_LOOP_TYPE(index, num_threads, index_t) {
    const index_t b = index / (HD * Q);
    const index_t hd = (index / Q) % HD;
    const index_t q = index % Q;

    for (index_t c = 0; c < C; c++) {
      scalar_t sum = 0;
      for (index_t d = 0; d < D; d++) {
        // Get image coordinates in pixel space.
        const scalar_t ix = grid_sampler_compute_source_index(samples[b][q][d][0], W);
        const scalar_t iy = grid_sampler_compute_source_index(samples[b][q][d][1], H);

        // Get corner pixel indices (referenced using compass directions).
        const index_t ix_nw = static_cast<index_t>(::floor(ix));
        const index_t iy_nw = static_cast<index_t>(::floor(iy));
        const index_t ix_ne = ix_nw + 1;
        const index_t iy_ne = iy_nw;
        const index_t ix_sw = ix_nw;
        const index_t iy_sw = iy_nw + 1;
        const index_t ix_se = ix_nw + 1;
        const index_t iy_se = iy_nw + 1;

        // Compute interpolation weights.
        const scalar_t nw = (ix_se - ix) * (iy_se - iy);
        const scalar_t ne = (ix - ix_sw) * (iy_sw - iy);
        const scalar_t sw = (ix_ne - ix) * (iy - iy_ne);
        const scalar_t se = (ix - ix_nw) * (iy - iy_nw);

        // Compute the sum.
        const scalar_t weight = weights[b][hd][q][d];
        if (within_bounds_2d(iy_nw, ix_nw, H, W)) {
          sum += images[b][c][iy_nw][ix_nw] * nw * weight;
        }
        if (within_bounds_2d(iy_ne, ix_ne, H, W)) {
          sum += images[b][c][iy_ne][ix_ne] * ne * weight;
        }
        if (within_bounds_2d(iy_sw, ix_sw, H, W)) {
          sum += images[b][c][iy_sw][ix_sw] * sw * weight;
        }
        if (within_bounds_2d(iy_se, ix_se, H, W)) {
          sum += images[b][c][iy_se][ix_se] * se * weight;
        }
      }
      outputs[b][hd][q][c] = sum;
    }
  }
}

template <typename scalar_t, typename index_t>
__launch_bounds__(256) __global__ void sample_sum_backward_kernel(
    const index_t num_threads,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits>
        output_gradients,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> images,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> samples,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> weights,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits>
        image_gradients,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits>
        sample_gradients,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits>
        weight_gradients) {
  // Extract dimensions.
  const index_t B = images.size(0);
  const index_t C = images.size(1);
  const index_t H = images.size(2);
  const index_t W = images.size(3);
  const index_t HD = weights.size(1);
  const index_t Q = weights.size(2);
  const index_t D = weights.size(3);

  CUDA_KERNEL_LOOP_TYPE(index, num_threads, index_t) {
    const index_t b = index / (HD * Q);
    const index_t hd = (index / Q) % HD;
    const index_t q = index % Q;

    const scalar_t scaling_x = 0.5 * static_cast<scalar_t>(W);
    const scalar_t scaling_y = 0.5 * static_cast<scalar_t>(H);

    for (index_t d = 0; d < D; d++) {
      const scalar_t weight = weights[b][hd][q][d];
      scalar_t weight_gradient = 0;

      for (index_t c = 0; c < C; c++) {
        const scalar_t output_gradient = output_gradients[b][hd][q][c];
        scalar_t sample_gradient_x = 0;
        scalar_t sample_gradient_y = 0;

        // Get image coordinates in pixel space.
        const scalar_t ix = grid_sampler_compute_source_index(samples[b][q][d][0], W);
        const scalar_t iy = grid_sampler_compute_source_index(samples[b][q][d][1], H);

        // Get corner pixel indices (referenced using compass directions).
        const index_t ix_nw = static_cast<index_t>(::floor(ix));
        const index_t iy_nw = static_cast<index_t>(::floor(iy));
        const index_t ix_ne = ix_nw + 1;
        const index_t iy_ne = iy_nw;
        const index_t ix_sw = ix_nw;
        const index_t iy_sw = iy_nw + 1;
        const index_t ix_se = ix_nw + 1;
        const index_t iy_se = iy_nw + 1;

        // Compute interpolation weights.
        const scalar_t nw = (ix_se - ix) * (iy_se - iy);
        const scalar_t ne = (ix - ix_sw) * (iy_sw - iy);
        const scalar_t sw = (ix_ne - ix) * (iy - iy_ne);
        const scalar_t se = (ix - ix_nw) * (iy - iy_nw);

        // Compute the sum.
        if (within_bounds_2d(iy_nw, ix_nw, H, W)) {
          const scalar_t pixel_nw = images[b][c][iy_nw][ix_nw];
          weight_gradient += output_gradient * nw * pixel_nw;
          const scalar_t pixel_nw_weight = pixel_nw * weight;
          sample_gradient_x -= pixel_nw_weight * (iy_se - iy);
          sample_gradient_y -= pixel_nw_weight * (ix_se - ix);
          atomicAdd(&image_gradients[b][c][iy_nw][ix_nw],
                    output_gradient * nw * weight);
        }
        if (within_bounds_2d(iy_ne, ix_ne, H, W)) {
          const scalar_t pixel_ne = images[b][c][iy_ne][ix_ne];
          weight_gradient += output_gradient * ne * pixel_ne;
          const scalar_t pixel_ne_weight = pixel_ne * weight;
          sample_gradient_x += pixel_ne_weight * (iy_sw - iy);
          sample_gradient_y -= pixel_ne_weight * (ix - ix_sw);
          atomicAdd(&image_gradients[b][c][iy_ne][ix_ne],
                    output_gradient * ne * weight);
        }
        if (within_bounds_2d(iy_sw, ix_sw, H, W)) {
          const scalar_t pixel_sw = images[b][c][iy_sw][ix_sw];
          weight_gradient += output_gradient * sw * pixel_sw;
          const scalar_t pixel_sw_weight = pixel_sw * weight;
          sample_gradient_x -= pixel_sw_weight * (iy - iy_ne);
          sample_gradient_y += pixel_sw_weight * (ix_ne - ix);
          atomicAdd(&image_gradients[b][c][iy_sw][ix_sw],
                    output_gradient * sw * weight);
        }
        if (within_bounds_2d(iy_se, ix_se, H, W)) {
          const scalar_t pixel_se = images[b][c][iy_se][ix_se];
          weight_gradient += output_gradient * se * pixel_se;
          const scalar_t pixel_se_weight = pixel_se * weight;
          sample_gradient_x += pixel_se_weight * (iy - iy_nw);
          sample_gradient_y += pixel_se_weight * (ix - ix_nw);
          atomicAdd(&image_gradients[b][c][iy_se][ix_se],
                    output_gradient * se * weight);
        }

        // Add to the sample gradients.
        atomicAdd(&sample_gradients[b][q][d][0],
                  output_gradient * sample_gradient_x * scaling_x);
        atomicAdd(&sample_gradients[b][q][d][1],
                  output_gradient * sample_gradient_y * scaling_y);
      }

      atomicAdd(&weight_gradients[b][hd][q][d], weight_gradient);
    }
  }
}

}  // namespace fused_grid_sum