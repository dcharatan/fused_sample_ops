#include <torch/extension.h>

#include "fused_grid_sum.cuh"
#include "sample_dot.cuh"
#include "sample_sum.cuh"

// TORCH_EXTENSION_NAME is defined by setup.py or CMakeLists.txt depending on
// the compilation method.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &fused_grid_sum::forward, "fused grid sum forward (CUDA)");
  m.def("backward", &fused_grid_sum::backward, "fused grid sum backward (CUDA)");
  m.def("sample_dot_forward", &fused_grid_sum::sample_dot_forward,
        "fused grid sample and dot product forward (CUDA)");
  m.def("sample_dot_backward", &fused_grid_sum::sample_dot_backward,
        "fused grid sample and dot product backward (CUDA)");
  m.def("sample_sum_forward", &fused_grid_sum::sample_sum_forward,
        "fused grid sample and sum forward (CUDA)");
  m.def("sample_sum_backward", &fused_grid_sum::sample_sum_backward,
        "fused grid sample and sum backward (CUDA)");
}
