cmake_minimum_required(VERSION 3.30)
project(xavier)

set(CMAKE_CXX_STANDARD 23)
# Metal-cpp
set(METAL_CPP "${CMAKE_SOURCE_DIR}/metal-cpp")
# Python and Pybind11
set(PYTHON "/opt/miniconda3/envs/tensor_x/include/python3.12")
set(PYBIND "/opt/miniconda3/envs/tensor_x/include/pybind11")
set(CMAKE_CXX_FLAGS "-undefined dynamic_lookup")
include_directories(${PYTHON})
include_directories(${PYBIND})
find_package(pybind11 REQUIRED)

set(HEADER_FILES
    core/dtype.h
    core/shape.h
    core/array.h
    core/iter.h
    core/ops.h
    graph/graph.h
    graph//mtl_graph.h
    backend/metal/metal.h
    backend/metal/mtl_context.h
    backend/metal/mtl_kernel.h
    backend/metal/mtl_kernels.h
)

set(SRC_FILES
    core/array.cpp
    graph/graph.cpp
    graph/mtl_graph.cpp
    backend/metal/mtl_context.cpp
    backend/metal/mtl_kernels.cpp
)

add_subdirectory(backend/metal)
# pybind11_add_module(${PROJECT_NAME} bind.cpp)
add_executable(${PROJECT_NAME} main.cpp ${SRC_FILES} ${HEADER_FILES})

# Add include directories
target_include_directories(${PROJECT_NAME} PRIVATE ${METAL_CPP})

# Metal cpp library (linker)
target_link_libraries(${PROJECT_NAME} PRIVATE
    "-framework Metal"
    "-framework Foundation"
    "-framework QuartzCore"
)