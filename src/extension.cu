#include <torch/extension.h>

#include "lines.cuh"
#include "points.cuh"

// TORCH_EXTENSION_NAME is defined by setup.py or CMakeLists.txt depending on
// the compilation method.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("render_points", &torchdraw_cuda::render_points, "render points (CUDA)");
  m.def("render_lines", &torchdraw_cuda::render_lines, "render lines (CUDA)");
}
