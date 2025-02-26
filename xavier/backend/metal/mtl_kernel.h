#pragma once

#include "metal.h"
#include "../../core/array.h"
#include "../../core/dtype.h"

namespace xv::backend::metal
{
    using namespace xv::core;

    struct MTLKernel : public std::enable_shared_from_this<MTLKernel>
    {
    private:
        // shared ptr gets released once kernel is released
        NS::SharedPtr<MTL::ComputePipelineState> state;
        Dtype dtype;

    public:
        MTLKernel(NS::SharedPtr<MTL::ComputePipelineState> state, Dtype dtype) : state(state), dtype(dtype)
        {
        }

        NS::SharedPtr<MTL::ComputePipelineState> get_state() { return state; }
        Dtype get_dtype() { return dtype; }
    };
}