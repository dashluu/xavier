#pragma once

#include "mtl_utils.h"

namespace xv::backend::metal
{
    void reduce(const std::string &name, std::shared_ptr<Array> input, std::shared_ptr<Array> output, MTLContext &ctx);
}