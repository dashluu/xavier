#pragma once

#include "mtl_utils.h"

namespace xv::backend::metal
{
    void full(ArrayPtr arr, int c, usize size, MTLContext &ctx);
    void arange(ArrayPtr arr, int start, int step, MTLContext &ctx);
}