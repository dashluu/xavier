#pragma once

#include "mtl_utils.h"

namespace xv::backend::metal
{
    void unary_ss(const std::string &name, std::shared_ptr<Array> input, std::shared_ptr<Array> output, MTLContext &ctx);
    void sparse_unary_ss(const std::string &name, std::shared_ptr<Array> input, std::shared_ptr<Array> output, MTLContext &ctx);
}