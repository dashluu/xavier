#pragma once

#include "mtl_context.h"

namespace xv::backend::metal
{
    template <class T1, class T2>
    inline std::vector<T2> v64to32(const std::vector<T1> &v)
    {
        std::vector<T2> v32(v.size());
        for (size_t i = 0; i < v.size(); i++)
        {
            v32[i] = static_cast<T2>(v[i]);
        }
        return v32;
    }

    inline std::vector<uint32_t> get_mtl_view(const ShapeView &view)
    {
        return v64to32<usize, uint32_t>(view);
    }

    inline std::vector<int32_t> get_mtl_stride(const ShapeStride &stride)
    {
        return v64to32<isize, int32_t>(stride);
    }

    inline std::vector<uint32_t> get_mtl_offsets(const std::vector<ArrayPtr> &arrs)
    {
        std::vector<uint32_t> offsets;
        for (auto &arr : arrs)
        {
            offsets.emplace_back(arr->get_offset());
        }
        return offsets;
    }

    inline void encode_buffer(NS::SharedPtr<MTL::Device> device, MTL::ComputeCommandEncoder *encoder, const void *buff, usize size, uint32_t &buff_idx)
    {
        MTL::Buffer *mtl_buff = device->newBuffer(buff, size, MTL::ResourceStorageModeShared, nullptr);
        encoder->setBuffer(mtl_buff, 0, buff_idx++);
    }

    inline void encode_array(NS::SharedPtr<MTL::Device> device, MTL::ComputeCommandEncoder *encoder, ArrayPtr arr, uint32_t &buff_idx)
    {
        encode_buffer(device, encoder, arr->get_buff_ptr(), arr->get_buff_nbytes(), buff_idx);
    }

    inline void encode_offset(NS::SharedPtr<MTL::Device> device, MTL::ComputeCommandEncoder *encoder, const std::vector<ArrayPtr> &arrs, uint32_t &buff_idx)
    {
        auto offset = get_mtl_offsets(arrs);
        encode_buffer(device, encoder, offset.data(), vsize(offset), buff_idx);
    }

    inline void encode_view(NS::SharedPtr<MTL::Device> device, MTL::ComputeCommandEncoder *encoder, ArrayPtr arr, uint32_t &buff_idx)
    {
        auto view = get_mtl_view(arr->get_view());
        encode_buffer(device, encoder, view.data(), vsize(view), buff_idx);
    }

    inline void encode_stride(NS::SharedPtr<MTL::Device> device, MTL::ComputeCommandEncoder *encoder, ArrayPtr arr, uint32_t &buff_idx)
    {
        auto stride = get_mtl_stride(arr->get_stride());
        encode_buffer(device, encoder, stride.data(), vsize(stride), buff_idx);
    }

    inline void ss_dispatch(MTLContext &ctx, MTL::CommandBuffer *cmd_buff, MTL::ComputeCommandEncoder *encoder, const std::string &name, usize numel)
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