#include <torch/extension.h>

#include "fused_grid_sum.cuh"
#include "grid_sample_dot.cuh"

// TORCH_EXTENSION_NAME is defined by setup.py or CMakeLists.txt depending on
// the compilation method.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &fused_grid_sum::forward, "fused grid sum forward (CUDA)");
  m.def("backward", &fused_grid_sum::backward, "fused grid sum backward (CUDA)");
  m.def("grid_sample_dot_forward", &fused_grid_sum::grid_sample_dot_forward,
        "fused grid sample and dot product (CUDA)");
}
