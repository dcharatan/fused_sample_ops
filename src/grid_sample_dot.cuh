#pragma once

#include <torch/extension.h>

#include "common.cuh"

namespace fused_grid_sum {

void grid_sample_dot_forward(torch::Tensor images,
                             torch::Tensor samples,
                             torch::Tensor queries,
                             torch::Tensor outputs);

template <typename scalar_t, typename index_t>
__launch_bounds__(256) __global__ void grid_sample_dot_forward_kernel(
    const index_t num_threads,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> images,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> samples,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> queries,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> outputs) {
  // Extract dimensions.
  const index_t B = images.size(0);
  const index_t C = images.size(1);
  const index_t H = images.size(2);
  const index_t W = images.size(3);
  const index_t S = samples.size(1);

  CUDA_KERNEL_LOOP_TYPE(index, num_threads, index_t) {
    const index_t b = index / S;
    const index_t s = index % S;

    // get the corresponding images x, y co-ordinates from samples
    scalar_t x = samples[b][s][0];
    scalar_t y = samples[b][s][1];

    scalar_t ix = grid_sampler_compute_source_index(x, W);
    scalar_t iy = grid_sampler_compute_source_index(y, H);

    // get NE, NW, SE, SW pixel values from (x, y)
    index_t ix_nw = static_cast<index_t>(::floor(ix));
    index_t iy_nw = static_cast<index_t>(::floor(iy));
    index_t ix_ne = ix_nw + 1;
    index_t iy_ne = iy_nw;
    index_t ix_sw = ix_nw;
    index_t iy_sw = iy_nw + 1;
    index_t ix_se = ix_nw + 1;
    index_t iy_se = iy_nw + 1;

    // get surfaces to each neighbor:
    scalar_t nw = (ix_se - ix) * (iy_se - iy);
    scalar_t ne = (ix - ix_sw) * (iy_sw - iy);
    scalar_t sw = (ix_ne - ix) * (iy - iy_ne);
    scalar_t se = (ix - ix_nw) * (iy - iy_nw);

    // calculate bilinear weighted pixel value and set outputs pixel
    scalar_t dot_product = 0;
    for (index_t c = 0; c < C; c++) {
      const scalar_t query = queries[b][s][c];

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
    outputs[b][s] = dot_product;
  }
}

}  // namespace fused_grid_sum