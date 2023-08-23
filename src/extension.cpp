#include <torch/extension.h>

// Determine the module name based on the compilation method.
#ifdef COMPILED_WITH_CMAKE
#define MODULE_NAME libtorchdraw_cuda
#else
#define MODULE_NAME TORCH_EXTENSION_NAME
#endif

torch::Tensor render_lines(torch::Tensor samples) {
  return samples;
}

PYBIND11_MODULE(MODULE_NAME, m) {
  m.def("render_lines", &render_lines, "render lines (CUDA)");
}
