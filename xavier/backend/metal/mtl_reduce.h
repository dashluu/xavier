#pragma once

#include "mtl_utils.h"

namespace xv::backend::metal
{
    void reduce(const std::string &name, ArrayPtr input, ArrayPtr output, MTLContext &ctx);
}