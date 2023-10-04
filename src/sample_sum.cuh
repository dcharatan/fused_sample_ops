#pragma once

#include <torch/extension.h>

#include "common.cuh"

namespace fused_sample_ops {

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
__device__ scalar_t draw_sample(
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> &images,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> &samples,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> &weights,
    const index_t H,
    const index_t W,
    const index_t b,
    const index_t hd,
    const index_t q,
    const index_t d,
    const index_t c) {
  scalar_t sum = 0;

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

  return sum;
}

template <typename scalar_t, typename index_t>
__launch_bounds__(256) __global__ void sample_sum_forward_kernel(
    const index_t num_elements,
    const index_t sums_per_block,
    const index_t D_padded,
    const index_t initial_loads_per_thread,
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

  // Determine this thread's index in (B, HD, Q, C).
  const index_t threads_per_sum = D_padded / initial_loads_per_thread;
  const index_t index_of_sum_in_block = threadIdx.x / threads_per_sum;
  const index_t index_bhdqc = blockIdx.x * sums_per_block + index_of_sum_in_block;

  if (index_bhdqc < num_elements) {
    const index_t b = index_bhdqc / (HD * Q * C);
    const index_t hd = (index_bhdqc / (Q * C)) % HD;
    const index_t q = (index_bhdqc / C) % Q;
    const index_t c = index_bhdqc % C;

    // Compute the indices used for the parallel reduction.
    const index_t index_in_block = threadIdx.x;
    const index_t index_in_sum = index_in_block % threads_per_sum;

    // Draw the initial samples.
    scalar_t sum = 0;
    const index_t start_in_d = index_in_sum * initial_loads_per_thread;
    for (index_t i = 0; i < initial_loads_per_thread; i++) {
      const index_t d = start_in_d + i;
      if (d < D) {
        sum += draw_sample(images, samples, weights, H, W, b, hd, q, d, c);
      }
    }

    atomicAdd(&outputs[b][hd][q][c], sum);
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
    const index_t b = index / (Q * D);
    const index_t q = (index / D) % Q;
    const index_t d = index % D;

    const scalar_t scaling_x = 0.5 * static_cast<scalar_t>(W);
    const scalar_t scaling_y = 0.5 * static_cast<scalar_t>(H);

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

    const bool nw_in_bounds = within_bounds_2d(iy_nw, ix_nw, H, W);
    const bool ne_in_bounds = within_bounds_2d(iy_ne, ix_ne, H, W);
    const bool sw_in_bounds = within_bounds_2d(iy_sw, ix_sw, H, W);
    const bool se_in_bounds = within_bounds_2d(iy_se, ix_se, H, W);

    scalar_t sample_gradient_x = 0;
    scalar_t sample_gradient_y = 0;

    for (index_t c = 0; c < C; c++) {
      const scalar_t pixel_nw = nw_in_bounds ? images[b][c][iy_nw][ix_nw] : 0;
      const scalar_t pixel_ne = ne_in_bounds ? images[b][c][iy_ne][ix_ne] : 0;
      const scalar_t pixel_sw = sw_in_bounds ? images[b][c][iy_sw][ix_sw] : 0;
      const scalar_t pixel_se = se_in_bounds ? images[b][c][iy_se][ix_se] : 0;

      scalar_t image_gradient_nw = 0;
      scalar_t image_gradient_ne = 0;
      scalar_t image_gradient_sw = 0;
      scalar_t image_gradient_se = 0;

      for (index_t hd = 0; hd < HD; hd++) {
        const scalar_t output_gradient = output_gradients[b][hd][q][c];
        const scalar_t weight = weights[b][hd][q][d];

        scalar_t weight_gradient = 0;

        // Compute the sum.
        if (nw_in_bounds) {
          weight_gradient += output_gradient * nw * pixel_nw;
          const scalar_t multiplier_nw = output_gradient * pixel_nw * weight;
          sample_gradient_x -= multiplier_nw * (iy_se - iy);
          sample_gradient_y -= multiplier_nw * (ix_se - ix);
          image_gradient_nw += output_gradient * weight;
        }
        if (ne_in_bounds) {
          weight_gradient += output_gradient * ne * pixel_ne;
          const scalar_t multiplier_ne = output_gradient * pixel_ne * weight;
          sample_gradient_x += multiplier_ne * (iy_sw - iy);
          sample_gradient_y -= multiplier_ne * (ix - ix_sw);
          image_gradient_ne += output_gradient * weight;
        }
        if (sw_in_bounds) {
          weight_gradient += output_gradient * sw * pixel_sw;
          const scalar_t multiplier_sw = output_gradient * pixel_sw * weight;
          sample_gradient_x -= multiplier_sw * (iy - iy_ne);
          sample_gradient_y += multiplier_sw * (ix_ne - ix);
          image_gradient_sw += output_gradient * weight;
        }
        if (se_in_bounds) {
          weight_gradient += output_gradient * se * pixel_se;
          const scalar_t multiplier_se = output_gradient * pixel_se * weight;
          sample_gradient_x += multiplier_se * (iy - iy_nw);
          sample_gradient_y += multiplier_se * (ix - ix_nw);
          image_gradient_se += output_gradient * weight;
        }

        // The loop ordering isn't ideal for the weight gradients (with a different one,
        // we would be able to write out the weight gradients in one operation). Note
        // that atomicAdd isn't needed here because this thread is the only one
        // modifying this entry.
        weight_gradients[b][hd][q][d] += weight_gradient;
      }

      if (nw_in_bounds) {
        atomicAdd(&image_gradients[b][c][iy_nw][ix_nw], image_gradient_nw * nw);
      }
      if (ne_in_bounds) {
        atomicAdd(&image_gradients[b][c][iy_ne][ix_ne], image_gradient_ne * ne);
      }
      if (sw_in_bounds) {
        atomicAdd(&image_gradients[b][c][iy_sw][ix_sw], image_gradient_sw * sw);
      }
      if (se_in_bounds) {
        atomicAdd(&image_gradients[b][c][iy_se][ix_se], image_gradient_se * se);
      }
    }

    // Set the sample gradients.
    sample_gradients[b][q][d][0] = sample_gradient_x * scaling_x;
    sample_gradients[b][q][d][1] = sample_gradient_y * scaling_y;
  }
}

}  // namespace fused_sample_ops