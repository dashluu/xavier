#pragma once

#include "mtl_command_encoder.h"

namespace xv::backend::metal
{
    void reduce_all(const std::string &name, ArrayPtr input, ArrayPtr output, std::shared_ptr<MTLContext> ctx);
    void reduce_col(const std::string &name, ArrayPtr input, ArrayPtr output, std::shared_ptr<MTLContext> ctx);
}