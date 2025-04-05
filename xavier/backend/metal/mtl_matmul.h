#pragma once

#include "mtl_command_encoder.h"

#define X_THREADS_PER_GROUP 8
#define Y_THREADS_PER_GROUP 8
#define Z_THREADS_PER_GROUP 4

namespace xv::backend::metal
{
    void matmul(ArrayPtr lhs, ArrayPtr rhs, ArrayPtr output, std::shared_ptr<MTLContext> ctx);
}