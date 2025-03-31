#pragma once

#include "mtl_utils.h"

namespace xv::backend::metal
{
    void copy(ArrayPtr src, ArrayPtr dst, MTLContext &ctx);
    void strided_copy(ArrayPtr src, ArrayPtr dst, MTLContext &ctx);
}