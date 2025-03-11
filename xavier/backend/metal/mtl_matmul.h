#pragma once

#include "mtl_utils.h"

namespace xv::backend::metal
{
    void matmul2d(std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs, std::shared_ptr<Array> output, MTLContext &ctx);
}