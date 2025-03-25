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
    graph/graph.h
)

set(SRC_FILES
    core/ops.cpp
    core/array.cpp
)

if(APPLE)
    SET(MTL_HEADER_FILES
        graph/mtl_graph.h
        backend/metal/metal.h
        backend/metal/mtl_context.h
        backend/metal/mtl_kernel.h
        backend/metal/mtl_utils.h
        backend/metal/mtl_initializers.h
        backend/metal/mtl_copy.h
        backend/metal/mtl_unary.h
        backend/metal/mtl_binary.h
        backend/metal/mtl_matmul.h
    )
    SET(MTL_SRC_FILES
        graph/mtl_graph.cpp
        backend/metal/mtl_context.cpp
        backend/metal/mtl_initializers.cpp
        backend/metal/mtl_copy.cpp
        backend/metal/mtl_unary.cpp
        backend/metal/mtl_binary.cpp
        backend/metal/mtl_matmul.cpp
    )
    add_subdirectory(backend/metal)
    pybind11_add_module(${PROJECT_NAME} bind.cpp ${SRC_FILES} ${HEADER_FILES} ${MTL_SRC_FILES} ${MTL_HEADER_FILES})
    # set(CMAKE_BUILD_TYPE Debug)
    # add_executable(${PROJECT_NAME} main.cpp ${SRC_FILES} ${HEADER_FILES} ${MTL_SRC_FILES} ${MTL_HEADER_FILES})
    # Metal-cpp
    set(MTL_CPP "${CMAKE_SOURCE_DIR}/metal-cpp")
    # Add include directories
    target_include_directories(${PROJECT_NAME} PRIVATE ${MTL_CPP})
    # Metal cpp library (linker)
    target_link_libraries(${PROJECT_NAME} PRIVATE
        "-framework Metal"
        "-framework Foundation"
        "-framework QuartzCore"
    )
endif()
