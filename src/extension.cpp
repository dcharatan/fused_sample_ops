#include <torch/extension.h>

torch::Tensor render_lines(torch::Tensor samples) {
  return samples;
}

// TORCH_EXTENSION_NAME is defined by setup.py or CMakeLists.txt depending on the compilation method.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("render_lines", &render_lines, "render lines (CUDA)");
}
