#pragma once

#include "mtl_command_encoder.h"

namespace xv::backend::metal
{
    void unary_ss(const std::string &name, ArrayPtr input, ArrayPtr output, std::shared_ptr<MTLContext> ctx);
}