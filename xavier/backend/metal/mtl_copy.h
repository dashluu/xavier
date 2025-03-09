#pragma once

#include "mtl_utils.h"

namespace xv::backend::metal
{
    void copy(std::shared_ptr<Array> input, std::shared_ptr<Array> output, MTLContext &ctx);
    void sparse_copy(std::shared_ptr<Array> input, std::shared_ptr<Array> output, MTLContext &ctx);
}