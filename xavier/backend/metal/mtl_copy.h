#pragma once

#include "mtl_utils.h"

namespace xv::backend::metal
{
    void copy(std::shared_ptr<Array> src, std::shared_ptr<Array> dst, MTLContext &ctx);
    void strided_copy(std::shared_ptr<Array> src, std::shared_ptr<Array> dst, MTLContext &ctx);
}