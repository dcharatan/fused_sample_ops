#pragma once

#include <torch/extension.h>

#include "common.cuh"

namespace fused_grid_sum {

void sample_dot_forward(torch::Tensor images,
                        torch::Tensor samples,
                        torch::Tensor queries,
                        torch::Tensor depths,
                        torch::Tensor outputs);

void sample_dot_backward(torch::Tensor output_gradients,
                         torch::Tensor images,
                         torch::Tensor samples,
                         torch::Tensor queries,
                         torch::Tensor depths,
                         torch::Tensor image_gradients,
                         torch::Tensor sample_gradients,
                         torch::Tensor query_gradients,
                         torch::Tensor depth_gradients);

template <typename scalar_t, typename index_t>
__launch_bounds__(256) __global__ void sample_dot_forward_kernel(
    const index_t num_threads,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> images,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> samples,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> queries,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> depths,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> outputs) {
  // Extract dimensions.
  const index_t B = images.size(0);
  const index_t C_images = images.size(1);
  const index_t H = images.size(2);
  const index_t W = images.size(3);
  const index_t Q = samples.size(1);
  const index_t D = samples.size(2);
  const index_t C_queries = queries.size(2);

  CUDA_KERNEL_LOOP_TYPE(index, num_threads, index_t) {
    const index_t b = index / (Q * D);
    const index_t q = (index / D) % Q;
    const index_t d = index % D;

    // Get image coordinates in pixel space.
    scalar_t ix = grid_sampler_compute_source_index(samples[b][q][d][0], W);
    scalar_t iy = grid_sampler_compute_source_index(samples[b][q][d][1], H);

    // Get corner pixel indices (referenced using compass directions).
    index_t ix_nw = static_cast<index_t>(::floor(ix));
    index_t iy_nw = static_cast<index_t>(::floor(iy));
    index_t ix_ne = ix_nw + 1;
    index_t iy_ne = iy_nw;
    index_t ix_sw = ix_nw;
    index_t iy_sw = iy_nw + 1;
    index_t ix_se = ix_nw + 1;
    index_t iy_se = iy_nw + 1;

    // Compute interpolation weights.
    scalar_t nw = (ix_se - ix) * (iy_se - iy);
    scalar_t ne = (ix - ix_sw) * (iy_sw - iy);
    scalar_t sw = (ix_ne - ix) * (iy - iy_ne);
    scalar_t se = (ix - ix_nw) * (iy - iy_nw);

    // Compute the dot product with respect to the image.
    scalar_t dot_product = 0;
    index_t c = 0;
    for (; c < C_queries; c++) {
      const scalar_t query = queries[b][q][c];

      if (within_bounds_2d(iy_nw, ix_nw, H, W)) {
        dot_product += images[b][c][iy_nw][ix_nw] * nw * query;
      }
      if (within_bounds_2d(iy_ne, ix_ne, H, W)) {
        dot_product += images[b][c][iy_ne][ix_ne] * ne * query;
      }
      if (within_bounds_2d(iy_sw, ix_sw, H, W)) {
        dot_product += images[b][c][iy_sw][ix_sw] * sw * query;
      }
      if (within_bounds_2d(iy_se, ix_se, H, W)) {
        dot_product += images[b][c][iy_se][ix_se] * se * query;
      }
    }

    // Compute the dot product with respect to the depth encoding.
    const scalar_t depth = depths[b][q][d];
    constexpr scalar_t PI = 3.141592654;
    scalar_t frequency = 2 * PI;
    bool use_cos = false;
    for (; c < C_images; c++) {
      // Add a positional encoding channel to the dot product.
      const scalar_t phase = use_cos ? PI * 0.5 : 0;
      const scalar_t query = sin(depth * frequency + phase);

      if (within_bounds_2d(iy_nw, ix_nw, H, W)) {
        dot_product += images[b][c][iy_nw][ix_nw] * nw * query;
      }
      if (within_bounds_2d(iy_ne, ix_ne, H, W)) {
        dot_product += images[b][c][iy_ne][ix_ne] * ne * query;
      }
      if (within_bounds_2d(iy_sw, ix_sw, H, W)) {
        dot_product += images[b][c][iy_sw][ix_sw] * sw * query;
      }
      if (within_bounds_2d(iy_se, ix_se, H, W)) {
        dot_product += images[b][c][iy_se][ix_se] * se * query;
      }

      // Update the positional encoding parameters.
      if (use_cos) {
        frequency *= 2;
      }
      use_cos = !use_cos;
    }

    outputs[b][q][d] = dot_product;
  }
}

template <typename scalar_t, typename index_t>
__launch_bounds__(256) __global__ void sample_dot_backward_kernel(
    const index_t num_threads,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits>
        output_gradients,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> images,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> samples,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> queries,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> depths,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits>
        image_gradients,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits>
        sample_gradients,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits>
        query_gradients,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits>
        depth_gradients) {
  // Extract dimensions.
  const index_t B = images.size(0);
  const index_t C_images = images.size(1);
  const index_t H = images.size(2);
  const index_t W = images.size(3);
  const index_t Q = samples.size(1);
  const index_t D = samples.size(2);
  const index_t C_queries = queries.size(2);

  CUDA_KERNEL_LOOP_TYPE(index, num_threads, index_t) {
    const index_t b = index / (Q * D);
    const index_t q = (index / D) % Q;
    const index_t d = index % D;

    scalar_t sample_gradient_x = 0;
    scalar_t sample_gradient_y = 0;

    // Get image coordinates in pixel space.
    scalar_t ix = grid_sampler_compute_source_index(samples[b][q][d][0], W);
    scalar_t iy = grid_sampler_compute_source_index(samples[b][q][d][1], H);

    // Get corner pixel indices (referenced using compass directions).
    index_t ix_nw = static_cast<index_t>(::floor(ix));
    index_t iy_nw = static_cast<index_t>(::floor(iy));
    index_t ix_ne = ix_nw + 1;
    index_t iy_ne = iy_nw;
    index_t ix_sw = ix_nw;
    index_t iy_sw = iy_nw + 1;
    index_t ix_se = ix_nw + 1;
    index_t iy_se = iy_nw + 1;

    // Compute interpolation weights.
    scalar_t nw = (ix_se - ix) * (iy_se - iy);
    scalar_t ne = (ix - ix_sw) * (iy_sw - iy);
    scalar_t sw = (ix_ne - ix) * (iy - iy_ne);
    scalar_t se = (ix - ix_nw) * (iy - iy_nw);

    // Accumulate image gradients and compute query gradients.
    const scalar_t output_gradient = output_gradients[b][q][d];
    index_t c = 0;
    for (; c < C_queries; c++) {
      const scalar_t query = queries[b][q][c];
      scalar_t query_gradient = 0;

      if (within_bounds_2d(iy_nw, ix_nw, H, W)) {
        const scalar_t pixel_nw = images[b][c][iy_nw][ix_nw];
        query_gradient += pixel_nw * nw;
        atomicAdd(&image_gradients[b][c][iy_nw][ix_nw], output_gradient * nw * query);
        const scalar_t pixel_nw_query = pixel_nw * query;
        sample_gradient_x -= pixel_nw_query * (iy_se - iy);
        sample_gradient_y -= pixel_nw_query * (ix_se - ix);
      }
      if (within_bounds_2d(iy_ne, ix_ne, H, W)) {
        const scalar_t pixel_ne = images[b][c][iy_ne][ix_ne];
        query_gradient += pixel_ne * ne;
        atomicAdd(&image_gradients[b][c][iy_ne][ix_ne], output_gradient * ne * query);
        const scalar_t pixel_ne_query = pixel_ne * query;
        sample_gradient_x += pixel_ne_query * (iy_sw - iy);
        sample_gradient_y -= pixel_ne_query * (ix - ix_sw);
      }
      if (within_bounds_2d(iy_sw, ix_sw, H, W)) {
        const scalar_t pixel_sw = images[b][c][iy_sw][ix_sw];
        query_gradient += pixel_sw * sw;
        atomicAdd(&image_gradients[b][c][iy_sw][ix_sw], output_gradient * sw * query);
        const scalar_t pixel_sw_query = pixel_sw * query;
        sample_gradient_x -= pixel_sw_query * (iy - iy_ne);
        sample_gradient_y += pixel_sw_query * (ix_ne - ix);
      }
      if (within_bounds_2d(iy_se, ix_se, H, W)) {
        const scalar_t pixel_se = images[b][c][iy_se][ix_se];
        query_gradient += pixel_se * se;
        atomicAdd(&image_gradients[b][c][iy_se][ix_se], output_gradient * se * query);
        const scalar_t pixel_se_query = pixel_se * query;
        sample_gradient_x += pixel_se_query * (iy - iy_nw);
        sample_gradient_y += pixel_se_query * (ix - ix_nw);
      }

      atomicAdd(&query_gradients[b][q][c], output_gradient * query_gradient);
    }

    // Accumulate image gradients and compute depth gradients.
    const scalar_t depth = depths[b][q][d];
    scalar_t depth_gradient = 0;
    constexpr scalar_t PI = 3.141592654;
    scalar_t frequency = 2 * PI;
    bool use_cos = false;
    for (; c < C_images; c++) {
      // Add a positional encoding channel to the dot product.
      const scalar_t phase = use_cos ? PI * 0.5 : 0;
      const scalar_t query = sin(depth * frequency + phase);
      const scalar_t query_gradient = frequency * cos(depth * frequency + phase);

      if (within_bounds_2d(iy_nw, ix_nw, H, W)) {
        const scalar_t pixel_nw = images[b][c][iy_nw][ix_nw];
        depth_gradient += nw * pixel_nw * query_gradient;
        atomicAdd(&image_gradients[b][c][iy_nw][ix_nw], output_gradient * nw * query);
        const scalar_t pixel_nw_query = pixel_nw * query;
        sample_gradient_x -= pixel_nw_query * (iy_se - iy);
        sample_gradient_y -= pixel_nw_query * (ix_se - ix);
      }
      if (within_bounds_2d(iy_ne, ix_ne, H, W)) {
        const scalar_t pixel_ne = images[b][c][iy_ne][ix_ne];
        depth_gradient += ne * pixel_ne * query_gradient;
        atomicAdd(&image_gradients[b][c][iy_ne][ix_ne], output_gradient * ne * query);
        const scalar_t pixel_ne_query = pixel_ne * query;
        sample_gradient_x += pixel_ne_query * (iy_sw - iy);
        sample_gradient_y -= pixel_ne_query * (ix - ix_sw);
      }
      if (within_bounds_2d(iy_sw, ix_sw, H, W)) {
        const scalar_t pixel_sw = images[b][c][iy_sw][ix_sw];
        depth_gradient += sw * pixel_sw * query_gradient;
        atomicAdd(&image_gradients[b][c][iy_sw][ix_sw], output_gradient * sw * query);
        const scalar_t pixel_sw_query = pixel_sw * query;
        sample_gradient_x -= pixel_sw_query * (iy - iy_ne);
        sample_gradient_y += pixel_sw_query * (ix_ne - ix);
      }
      if (within_bounds_2d(iy_se, ix_se, H, W)) {
        const scalar_t pixel_se = images[b][c][iy_se][ix_se];
        depth_gradient += se * pixel_se * query_gradient;
        atomicAdd(&image_gradients[b][c][iy_se][ix_se], output_gradient * se * query);
        const scalar_t pixel_se_query = pixel_se * query;
        sample_gradient_x += pixel_se_query * (iy - iy_nw);
        sample_gradient_y += pixel_se_query * (ix - ix_nw);
      }

      // Update the positional encoding parameters.
      if (use_cos) {
        frequency *= 2;
      }
      use_cos = !use_cos;
    }
    atomicAdd(&depth_gradients[b][q][d], output_gradient * depth_gradient);

    const scalar_t scaling_x = 0.5 * static_cast<scalar_t>(W);
    const scalar_t scaling_y = 0.5 * static_cast<scalar_t>(H);
    atomicAdd(&sample_gradients[b][q][d][0],
              output_gradient * sample_gradient_x * scaling_x);
    atomicAdd(&sample_gradients[b][q][d][1],
              output_gradient * sample_gradient_y * scaling_y);
  }
}

}  // namespace fused_grid_sum