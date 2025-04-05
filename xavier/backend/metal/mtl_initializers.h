#pragma once

#include "mtl_command_encoder.h"

namespace xv::backend::metal
{
    void full(ArrayPtr arr, int c, usize size, std::shared_ptr<MTLContext> ctx);
    void arange(ArrayPtr arr, int start, int step, std::shared_ptr<MTLContext> ctx);
}