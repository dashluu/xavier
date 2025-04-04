#pragma once

#include "mtl_utils.h"

namespace xv::backend::metal
{
    /**
     * @brief Performs binary operations on arrays with automatic stride detection
     *
     * @param name Base name of the Metal kernel (e.g., "add", "mul")
     * @param lhs Left-hand side input array
     * @param rhs Right-hand side input array
     * @param output Output array to store results
     * @param ctx Metal context containing device and command queue
     *
     * Buffer Layout:
     * 1. ndim (if strided): Number of dimensions
     * 2. offsets: LHS, RHS, and output buffer offsets
     * 3. view (if strided): Array dimensions
     * 4. stride (if strided input): LHS array strides
     * 5. stride (if strided input): RHS array strides
     * 6. stride (if strided output): Output array strides
     * 7. lhs array data
     * 8. rhs array data
     * 9. output array data
     *
     * Kernel Naming:
     * Format: {name}_{mode}_{dtype}
     * Where:
     * - mode: Two characters based on array contiguity
     *   First char: output ('s' if strided, 'v' if contiguous)
     *   Second char: input ('s' if either LHS or RHS is strided, 'v' if both contiguous)
     * - dtype: Data type of the arrays (e.g., "f32", "i32")
     *
     * Example:
     * add_sv_f32: Addition with strided output, contiguous inputs, float32 type
     */
    void binary_ss(const std::string &name, ArrayPtr lhs, ArrayPtr rhs, ArrayPtr output, MTLContext &ctx);
}