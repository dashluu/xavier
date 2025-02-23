project(metal_kernels)

set(CMAKE_CXX_STANDARD 14)

function(build_kernel KERNEL)
    set(SRCFILE ${CMAKE_CURRENT_SOURCE_DIR}/${KERNEL}.metal)
    # Extracts just the stem (filename without extension) from the KERNEL path
    cmake_path(GET KERNEL STEM TARGET)
    set(METAL_FLAGS -Wall -Wextra -fno-fast-math -gline-tables-only -frecord-sources)
    add_custom_command(
        COMMAND xcrun -sdk macosx metal
                    ${METAL_FLAGS}
                    -c ${SRCFILE}
                    -I ${PROJECT_SOURCE_DIR}
                    -o ${TARGET}.air
        DEPENDS ${SRCFILE} "${ARGN}"
        OUTPUT ${TARGET}.air
        COMMENT "Building ${TARGET}.air"
        VERBATIM
    )
    # Adds the generated .air file to a list named KERNEL_AIR in the parent scope
    set(KERNEL_AIR ${TARGET}.air ${KERNEL_AIR} PARENT_SCOPE)
endfunction(build_kernel)

build_kernel(utils utils.h)
build_kernel(binary utils.h)
build_kernel(initializers)
build_kernel(unary)
build_kernel(copy utils.h)

message(STATUS "Kernel AIR Files: ${KERNEL_AIR}")

add_custom_command(
    OUTPUT kernels.metallib
    COMMAND xcrun -sdk macosx metallib ${KERNEL_AIR} -o kernels.metallib
    DEPENDS ${KERNEL_AIR}
    COMMENT "Building kernels.metallib"
    VERBATIM
)

add_custom_target(
    metal_kernels ALL
    DEPENDS kernels.metallib
)