#include <torch/extension.h>

#include "fused_grid_sum.cuh"

// TORCH_EXTENSION_NAME is defined by setup.py or CMakeLists.txt depending on
// the compilation method.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &fused_grid_sum::forward, "fused grid sum forward (CUDA)");
}
