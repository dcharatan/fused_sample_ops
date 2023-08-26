#include <torch/extension.h>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

namespace torchdraw_cuda {

template <typename scalar_t, template <typename U> class PtrTraits, typename index_t>
__device__ glm::vec3 read_vec3(
    const torch::TensorAccessor<scalar_t, 1, PtrTraits, index_t> &accessor) {
  /// Extract a glm::vec3 from a one-dimensional accessor.
  glm::vec3 result;
  for (int i = 0; i < 3; i++) {
    result[i] = static_cast<float>(accessor[i]);
  }
  return result;
}

template <typename scalar_t, template <typename U> class PtrTraits, typename index_t>
__device__ glm::vec2 read_vec2(
    const torch::TensorAccessor<scalar_t, 1, PtrTraits, index_t> &accessor) {
  /// Extract a glm::vec2 from a one-dimensional accessor.
  glm::vec2 result;
  for (int i = 0; i < 2; i++) {
    result[i] = static_cast<float>(accessor[i]);
  }
  return result;
}

}  // namespace torchdraw_cuda