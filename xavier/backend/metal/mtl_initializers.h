#pragma once

#include "mtl_utils.h"

namespace xv::backend::metal
{
    void full(std::shared_ptr<Array> arr, float c, MTLContext &ctx);
    void arange(std::shared_ptr<Array> arr, int start, int step, MTLContext &ctx);
}