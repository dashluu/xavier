#pragma once

#include "mtl_command_encoder.h"

namespace xv::backend::metal
{
    void binary_ss(const std::string &name, ArrayPtr lhs, ArrayPtr rhs, ArrayPtr output, std::shared_ptr<MTLContext> ctx);
}