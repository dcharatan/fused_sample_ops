cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(torchdraw_cuda LANGUAGES CXX CUDA)

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

# Define the torchdraw_cuda library and link it to Torch and Python.
add_library(torchdraw_cuda SHARED
	src/extension.cpp
)
set_target_properties(torchdraw_cuda PROPERTIES CUDA_ARCHITECTURES "70;75;86")
target_link_libraries(torchdraw_cuda PRIVATE ${PYTHON_LIBRARIES} ${TORCH_LIBRARIES})
target_include_directories(torchdraw_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${Python_INCLUDE_DIRS})
