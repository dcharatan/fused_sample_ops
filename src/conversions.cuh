#pragma once

#include <torch/extension.h>
#include <glm/vec2.hpp>
#include "glm_adapter.cuh"

namespace fused_grid_sum {

template <typename scalar_t>
__device__ glm::vec2 pixel_to_world(
    const glm::vec2 &xy,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> &bounds,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits>
        &image_shape) {
  const glm::vec2 minima = read_vec2(bounds[0]);
  const glm::vec2 maxima = read_vec2(bounds[1]);
  const glm::vec2 image_wh{static_cast<float>(image_shape[1]),
                           static_cast<float>(image_shape[0])};
  return (xy / image_wh) * (maxima - minima) + minima;
}

template <typename scalar_t>
__device__ glm::vec2 world_to_pixel(
    const glm::vec2 &xy,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> &bounds,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits>
        &image_shape) {
  const glm::vec2 minima = read_vec2(bounds[0]);
  const glm::vec2 maxima = read_vec2(bounds[1]);
  const glm::vec2 image_wh{static_cast<float>(image_shape[1]),
                           static_cast<float>(image_shape[0])};
  return (xy - minima) / (maxima - minima) * image_wh;
}

}  // namespace fused_grid_sum