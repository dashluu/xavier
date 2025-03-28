#pragma once

#include "mtl_context.h"

namespace xv::backend::metal
{
    template <class T1, class T2>
    std::vector<T2> v64to32(const std::vector<T1> &v)
    {
        std::vector<T2> v32(v.size());
        for (size_t i = 0; i < v.size(); i++)
        {
            v32[i] = static_cast<T2>(v[i]);
        }
        return v32;
    }

    inline void ss_dispatch(MTLContext &ctx, MTL::CommandBuffer *cmd_buff, MTL::ComputeCommandEncoder *encoder, const std::string &name, uint64_t numel)
    {
        auto kernel = ctx.get_kernel(name);
        encoder->setComputePipelineState(kernel->get_state().get());
        auto grid_size = MTL::Size::Make(numel, 1, 1);
        auto threadgroup_size = MTL::Size::Make(kernel->get_state()->maxTotalThreadsPerThreadgroup(), 1, 1);
        encoder->dispatchThreads(grid_size, threadgroup_size);
        encoder->endEncoding();
        cmd_buff->commit();
        cmd_buff->waitUntilCompleted();
    }
}