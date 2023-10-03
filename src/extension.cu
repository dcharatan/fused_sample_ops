#include <torch/extension.h>

#include "sample_dot.cuh"
#include "sample_sum.cuh"

// TORCH_EXTENSION_NAME is defined by setup.py or CMakeLists.txt depending on
// the compilation method.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sample_dot_forward", &fused_sample_ops::sample_dot_forward,
        "fused grid sample and dot product forward (CUDA)");
  m.def("sample_dot_backward", &fused_sample_ops::sample_dot_backward,
        "fused grid sample and dot product backward (CUDA)");
  m.def("sample_sum_forward", &fused_sample_ops::sample_sum_forward,
        "fused grid sample and sum forward (CUDA)");
  m.def("sample_sum_backward", &fused_sample_ops::sample_sum_backward,
        "fused grid sample and sum backward (CUDA)");
}
