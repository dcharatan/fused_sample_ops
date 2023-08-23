#include <torch/extension.h>

torch::Tensor render_lines(torch::Tensor samples) {
  return samples;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("render_lines", &render_lines, "render lines (CUDA)");
}
