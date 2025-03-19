#pragma once

#include "mtl_utils.h"

namespace xv::backend::metal
{
    void binary_ss(const std::string &name, std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs, std::shared_ptr<Array> output, MTLContext &ctx);
    void sparse_binary_ss(const std::string &name, std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs, std::shared_ptr<Array> output, MTLContext &ctx);
}