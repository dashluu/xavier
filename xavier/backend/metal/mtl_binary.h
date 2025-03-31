#pragma once

#include "mtl_utils.h"

namespace xv::backend::metal
{
    void binary_ss(const std::string &name, ArrayPtr lhs, ArrayPtr rhs, ArrayPtr output, MTLContext &ctx);
    void strided_binary_ss(const std::string &name, ArrayPtr lhs, ArrayPtr rhs, ArrayPtr output, MTLContext &ctx);
}