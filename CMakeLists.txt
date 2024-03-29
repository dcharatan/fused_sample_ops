cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(fused_sample_ops LANGUAGES CXX CUDA)

# Connect the project to Python.
find_package(Python COMPONENTS Development)

# Connect the project to the installation of PyTorch that's in the virtual environment.
set(TORCH_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/venv/lib/python3.10/site-packages/torch")
find_package(Torch REQUIRED PATHS "${TORCH_ROOT}")

# Set compiler flags.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)

# Define the _cuda library.
add_library(_cuda SHARED
	src/extension.cu
	src/sample_dot.cu
	src/sample_sum.cu
)

# This workaround is needed to avoid linking errors (undefined symbols).
# See https://github.com/pytorch/pytorch/issues/38122#issuecomment-694203281
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

# Ensure that the name supplied to PYBIND11_MODULE is _cuda and that the output library produced by the CMake build is _cuda.so
add_definitions(-DTORCH_EXTENSION_NAME=_cuda)
set_target_properties(_cuda PROPERTIES PREFIX "")

target_link_libraries(_cuda PUBLIC ${PYTHON_LIBRARIES} ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
target_include_directories(_cuda PUBLIC src third_party/glm ${Python_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})
