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
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> queries,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> depths,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> outputs) {
  // Extract dimensions.
  const index_t B = images.size(0);
  const index_t C_images = images.size(1);
  const index_t H = images.size(2);
  const index_t W = images.size(3);
  const index_t Q = samples.size(1);
  const index_t D = samples.size(2);
  const index_t HD = queries.size(1);
  const index_t C_queries = queries.size(3);

  CUDA_KERNEL_LOOP_TYPE(index, num_threads, index_t) {
    const index_t b = index / (HD * Q * D);
    const index_t hd = (index / (Q * D)) % HD;
    const index_t q = (index / D) % Q;
    const index_t d = index % D;

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

    // Do bounds checks.
    const bool nw_in_bounds = within_bounds_2d(iy_nw, ix_nw, H, W);
    const bool ne_in_bounds = within_bounds_2d(iy_ne, ix_ne, H, W);
    const bool sw_in_bounds = within_bounds_2d(iy_sw, ix_sw, H, W);
    const bool se_in_bounds = within_bounds_2d(iy_se, ix_se, H, W);

    // Compute the dot product with respect to the image.
    scalar_t dot_product = 0;
    for (index_t c = 0; c < C_queries; c++) {
      const scalar_t query = queries[b][hd][q][c];

      if (nw_in_bounds) {
        dot_product += images[b][c][iy_nw][ix_nw] * nw * query;
      }
      if (ne_in_bounds) {
        dot_product += images[b][c][iy_ne][ix_ne] * ne * query;
      }
      if (sw_in_bounds) {
        dot_product += images[b][c][iy_sw][ix_sw] * sw * query;
      }
      if (se_in_bounds) {
        dot_product += images[b][c][iy_se][ix_se] * se * query;
      }
    }

    // Compute the dot product with respect to the depth encoding.
    const scalar_t depth = depths[b][q][d];
    constexpr scalar_t PI = 3.141592654;
    scalar_t frequency = 2 * PI;
    bool use_cos = false;
    for (index_t c = C_queries; c < C_images; c++) {
      // Add a positional encoding channel to the dot product.
      const scalar_t phase = use_cos ? PI * 0.5 : 0;
      const scalar_t query = sin(depth * frequency + phase);

      if (nw_in_bounds) {
        dot_product += images[b][c][iy_nw][ix_nw] * nw * query;
      }
      if (ne_in_bounds) {
        dot_product += images[b][c][iy_ne][ix_ne] * ne * query;
      }
      if (sw_in_bounds) {
        dot_product += images[b][c][iy_sw][ix_sw] * sw * query;
      }
      if (se_in_bounds) {
        dot_product += images[b][c][iy_se][ix_se] * se * query;
      }

      // Update the positional encoding parameters.
      if (use_cos) {
        frequency *= 2;
      }
      use_cos = !use_cos;

      outputs[b][hd][q][d] = dot_product;
    }
  }
}

template <typename scalar_t, typename index_t>
__launch_bounds__(256) __global__ void sample_dot_backward_kernel(
    const index_t num_threads,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits>
        output_gradients,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> images,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> samples,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> queries,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> depths,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits>
        image_gradients,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits>
        sample_gradients,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits>
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
  const index_t HD = queries.size(1);
  const index_t C_queries = queries.size(3);

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

    // Do bounds checks.
    const bool nw_in_bounds = within_bounds_2d(iy_nw, ix_nw, H, W);
    const bool ne_in_bounds = within_bounds_2d(iy_ne, ix_ne, H, W);
    const bool sw_in_bounds = within_bounds_2d(iy_sw, ix_sw, H, W);
    const bool se_in_bounds = within_bounds_2d(iy_se, ix_se, H, W);

    // Set up positional encoding counters.
    const scalar_t depth = depths[b][q][d];
    scalar_t depth_gradient = 0;
    constexpr scalar_t PI = 3.141592654;
    scalar_t frequency = 2 * PI;
    bool use_cos = false;

    // Accumulate image gradients and compute query gradients.
    for (index_t c = 0; c < C_images; c++) {
      const scalar_t pixel_nw = nw_in_bounds ? images[b][c][iy_nw][ix_nw] : 0;
      const scalar_t pixel_ne = ne_in_bounds ? images[b][c][iy_ne][ix_ne] : 0;
      const scalar_t pixel_sw = sw_in_bounds ? images[b][c][iy_sw][ix_sw] : 0;
      const scalar_t pixel_se = se_in_bounds ? images[b][c][iy_se][ix_se] : 0;

      scalar_t image_gradients_nw = 0;
      scalar_t image_gradients_ne = 0;
      scalar_t image_gradients_sw = 0;
      scalar_t image_gradients_se = 0;

      if (c < C_queries) {
        // Compute query gradients.
        for (index_t hd = 0; hd < HD; hd++) {
          const scalar_t output_gradient = output_gradients[b][hd][q][d];
          const scalar_t query = queries[b][hd][q][c];
          const scalar_t output_gradient_query = output_gradient * query;
          scalar_t query_gradient = 0;

          if (nw_in_bounds) {
            query_gradient += pixel_nw * nw;
            image_gradients_nw += output_gradient_query * nw;
            const scalar_t pixel_nw_query = output_gradient_query * pixel_nw;
            sample_gradient_x -= pixel_nw_query * (iy_se - iy);
            sample_gradient_y -= pixel_nw_query * (ix_se - ix);
          }
          if (ne_in_bounds) {
            query_gradient += pixel_ne * ne;
            image_gradients_ne += output_gradient_query * ne;
            const scalar_t pixel_ne_query = output_gradient_query * pixel_ne;
            sample_gradient_x += pixel_ne_query * (iy_sw - iy);
            sample_gradient_y -= pixel_ne_query * (ix - ix_sw);
          }
          if (sw_in_bounds) {
            query_gradient += pixel_sw * sw;
            image_gradients_sw += output_gradient_query * sw;
            const scalar_t pixel_sw_query = output_gradient_query * pixel_sw;
            sample_gradient_x -= pixel_sw_query * (iy - iy_ne);
            sample_gradient_y += pixel_sw_query * (ix_ne - ix);
          }
          if (se_in_bounds) {
            query_gradient += pixel_se * se;
            image_gradients_se += output_gradient_query * se;
            const scalar_t pixel_se_query = output_gradient_query * pixel_se;
            sample_gradient_x += pixel_se_query * (iy - iy_nw);
            sample_gradient_y += pixel_se_query * (ix - ix_nw);
          }

          atomicAdd(&query_gradients[b][hd][q][c], output_gradient * query_gradient);
        }
      } else {
        // Compute depth gradients.
        const scalar_t phase = use_cos ? PI * 0.5 : 0;
        const scalar_t query = sin(depth * frequency + phase);
        const scalar_t query_gradient = frequency * cos(depth * frequency + phase);

        for (index_t hd = 0; hd < HD; hd++) {
          const scalar_t output_gradient = output_gradients[b][hd][q][d];

          if (nw_in_bounds) {
            const scalar_t output_gradient_nw = output_gradient * nw;
            depth_gradient += output_gradient_nw * pixel_nw * query_gradient;
            image_gradients_nw += output_gradient_nw * query;
            const scalar_t pixel_nw_query = output_gradient * pixel_nw * query;
            sample_gradient_x -= pixel_nw_query * (iy_se - iy);
            sample_gradient_y -= pixel_nw_query * (ix_se - ix);
          }
          if (ne_in_bounds) {
            const scalar_t output_gradient_ne = output_gradient * ne;
            depth_gradient += output_gradient_ne * pixel_ne * query_gradient;
            image_gradients_ne += output_gradient_ne * query;
            const scalar_t pixel_ne_query = output_gradient * pixel_ne * query;
            sample_gradient_x += pixel_ne_query * (iy_sw - iy);
            sample_gradient_y -= pixel_ne_query * (ix - ix_sw);
          }
          if (sw_in_bounds) {
            const scalar_t output_gradient_sw = output_gradient * sw;
            depth_gradient += output_gradient_sw * pixel_sw * query_gradient;
            image_gradients_sw += output_gradient_sw * query;
            const scalar_t pixel_sw_query = output_gradient * pixel_sw * query;
            sample_gradient_x -= pixel_sw_query * (iy - iy_ne);
            sample_gradient_y += pixel_sw_query * (ix_ne - ix);
          }
          if (se_in_bounds) {
            const scalar_t output_gradient_se = output_gradient * se;
            depth_gradient += output_gradient_se * pixel_se * query_gradient;
            image_gradients_se += output_gradient_se * query;
            const scalar_t pixel_se_query = output_gradient * pixel_se * query;
            sample_gradient_x += pixel_se_query * (iy - iy_nw);
            sample_gradient_y += pixel_se_query * (ix - ix_nw);
          }
        }

        // Update the positional encoding parameters.
        if (use_cos) {
          frequency *= 2;
        }
        use_cos = !use_cos;
      }

      if (nw_in_bounds) {
        atomicAdd(&image_gradients[b][c][iy_nw][ix_nw], image_gradients_nw);
      }
      if (ne_in_bounds) {
        atomicAdd(&image_gradients[b][c][iy_ne][ix_ne], image_gradients_ne);
      }
      if (sw_in_bounds) {
        atomicAdd(&image_gradients[b][c][iy_sw][ix_sw], image_gradients_sw);
      }
      if (se_in_bounds) {
        atomicAdd(&image_gradients[b][c][iy_se][ix_se], image_gradients_se);
      }
    }

    depth_gradients[b][q][d] = depth_gradient;

    const scalar_t scaling_x = 0.5 * static_cast<scalar_t>(W);
    const scalar_t scaling_y = 0.5 * static_cast<scalar_t>(H);
    sample_gradients[b][q][d][0] = sample_gradient_x * scaling_x;
    sample_gradients[b][q][d][1] = sample_gradient_y * scaling_y;
  }
}

}  // namespace fused_grid_sum