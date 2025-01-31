cmake_minimum_required(VERSION 3.30)
project(xavier)

set(CMAKE_CXX_STANDARD 23)
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
    core/graph.h
)

set(SRC_FILES
    core/array.cpp
    core/graph.cpp
)

add_library(xv STATIC ${SRC_FILES} ${HEADER_FILES})
pybind11_add_module(xavier bind.cpp)
target_link_libraries(xavier PRIVATE xv)
add_subdirectory(backend/metal)