#pragma once

#include <torch/extension.h>

#include "common.cuh"

namespace fused_grid_sum {

void grid_sample_dot_forward(torch::Tensor images,
                             torch::Tensor samples,
                             torch::Tensor outputs);

template <typename scalar_t, typename index_t>
__launch_bounds__(256) __global__ void grid_sample_dot_forward_kernel(
    const index_t num_threads,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> images,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> samples,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> outputs) {
  index_t C = images.size(1);
  index_t inp_H = images.size(2);
  index_t inp_W = images.size(3);
  index_t out_H = samples.size(1);
  index_t out_W = samples.size(2);
  index_t inp_sN = images.stride(0);
  index_t inp_sC = images.stride(1);
  index_t inp_sH = images.stride(2);
  index_t inp_sW = images.stride(3);
  index_t grid_sN = samples.stride(0);
  index_t grid_sH = samples.stride(1);
  index_t grid_sW = samples.stride(2);
  index_t grid_sCoor = samples.stride(3);
  index_t out_sN = outputs.stride(0);
  index_t out_sC = outputs.stride(1);
  index_t out_sH = outputs.stride(2);
  index_t out_sW = outputs.stride(3);

  CUDA_KERNEL_LOOP_TYPE(index, num_threads, index_t) {
    const index_t w = index % out_W;
    const index_t h = (index / out_W) % out_H;
    const index_t n = index / (out_H * out_W);

    // get the corresponding images x, y co-ordinates from samples
    scalar_t x = samples[n][h][w][0];
    scalar_t y = samples[n][h][w][1];

    scalar_t ix = grid_sampler_compute_source_index(x, inp_W);
    scalar_t iy = grid_sampler_compute_source_index(y, inp_H);

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
    for (index_t c = 0; c < C; ++c) {
      scalar_t out_acc = 0;
      if (within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W)) {
        out_acc += images[n][c][iy_nw][ix_nw] * nw;
      }
      if (within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W)) {
        out_acc += images[n][c][iy_ne][ix_ne] * ne;
      }
      if (within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W)) {
        out_acc += images[n][c][iy_sw][ix_sw] * sw;
      }
      if (within_bounds_2d(iy_se, ix_se, inp_H, inp_W)) {
        out_acc += images[n][c][iy_se][ix_se] * se;
      }
      outputs[n][c][h][w] = out_acc;
    }
  }
}

}  // namespace fused_grid_sum