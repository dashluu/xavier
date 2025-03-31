#pragma once

#include "mtl_utils.h"

#define X_THREADS_PER_GROUP 8
#define Y_THREADS_PER_GROUP 8
#define Z_THREADS_PER_GROUP 4

namespace xv::backend::metal
{
    void matmul(ArrayPtr lhs, ArrayPtr rhs, ArrayPtr output, MTLContext &ctx);
    void strided_matmul(ArrayPtr lhs, ArrayPtr rhs, ArrayPtr output, MTLContext &ctx);
}