#pragma once

#include "mtl_utils.h"

namespace xv::backend::metal
{
    void unary_ss(const std::string &name, ArrayPtr input, ArrayPtr output, MTLContext &ctx);
    void strided_unary_ss(const std::string &name, ArrayPtr input, ArrayPtr output, MTLContext &ctx);
}