#pragma once

#include "mtl_utils.h"

namespace xv::backend::metal
{
    /**
     * @brief Performs unary operations on arrays with automatic stride detection
     *
     * @param name Base name of the Metal kernel (e.g., "exp", "log")
     * @param input Input array to perform operation on
     * @param output Output array to store results
     * @param ctx Metal context containing device and command queue
     *
     * Buffer Layout:
     * 1. ndim (if strided): Number of dimensions
     * 2. offsets: Input and output buffer offsets
     * 3. view (if strided): Array dimensions
     * 4. stride (if strided input): Input array strides
     * 5. stride (if strided output): Output array strides
     * 6. input array data
     * 7. output array data
     *
     * Kernel Naming:
     * Format: {name}_{mode}_{dtype}
     * Where:
     * - mode: Two characters based on array contiguity
     *   First char: output ('s' if strided, 'v' if contiguous)
     *   Second char: input ('s' if strided, 'v' if contiguous)
     * - dtype: Data type of the arrays (e.g., "f32", "i32")
     *
     * Example:
     * exp_sv_f32: Exponential operation with strided output, contiguous input, float32 type
     */
    void unary_ss(const std::string &name, ArrayPtr input, ArrayPtr output, MTLContext &ctx);
}