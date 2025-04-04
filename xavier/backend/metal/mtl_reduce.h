#pragma once

#include "mtl_utils.h"

namespace xv::backend::metal
{
    void reduce_all(const std::string &name, ArrayPtr input, ArrayPtr output, MTLContext &ctx);
    void reduce_col(const std::string &name, ArrayPtr input, ArrayPtr output, MTLContext &ctx);
}