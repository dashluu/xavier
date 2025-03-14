#pragma once

#include "mtl_utils.h"

#define X_THREADS_PER_GROUP 8
#define Y_THREADS_PER_GROUP 8

namespace xv::backend::metal
{
    void matmul2d(std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs, std::shared_ptr<Array> output, MTLContext &ctx);
}