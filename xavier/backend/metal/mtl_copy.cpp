#include "mtl_copy.h"

namespace xv::backend::metal
{
    void copy(ArrayPtr src, ArrayPtr dst, MTLContext &ctx)
    {
        auto cmd_queue = ctx.get_cmd_queue();
        auto cmd_buff = cmd_queue->commandBuffer();
        auto encoder = cmd_buff->computeCommandEncoder();
        auto device = ctx.get_device();
        // Offset
        auto offset = get_mtl_offsets({src, dst});
        auto offset_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(offset.data(), vsize(offset), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(offset_buff.get(), 0, 0);
        // src and dst buffers
        auto in_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(src->get_buff_ptr(), src->get_buff_nbytes(), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(in_buff.get(), 0, 1);
        auto out_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(dst->get_buff_ptr(), dst->get_buff_nbytes(), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(out_buff.get(), 0, 2);
        auto name = "copy_" + src->get_dtype().str();
        ss_dispatch(ctx, cmd_buff, encoder, name, src->get_numel());
    }

    void strided_copy(ArrayPtr src, ArrayPtr dst, MTLContext &ctx)
    {
        auto cmd_queue = ctx.get_cmd_queue();
        auto cmd_buff = cmd_queue->commandBuffer();
        auto encoder = cmd_buff->computeCommandEncoder();
        auto device = ctx.get_device();

        // # dimensions
        // Cast to 32-bit since the kernel accepts uint buffer
        auto ndim = static_cast<uint32_t>(src->get_ndim());
        auto ndim_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(&ndim, sizeof(uint32_t), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(ndim_buff.get(), 0, 0);

        // Offset
        auto offset = get_mtl_offsets({src, dst});
        auto offset_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(offset.data(), vsize(offset), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(offset_buff.get(), 0, 1);

        // View
        // Cast to 32-bit since the kernel accepts uint buffertr
        std::vector<uint32_t> view = get_mtl_view(src->get_view());
        auto view_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(view.data(), view.size() * sizeof(uint32_t), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(view_buff.get(), 0, 2);

        // Stride
        // Cast to 32-bit since the kernel accepts uint buffer
        std::vector<int32_t> stride = get_mtl_stride(src->get_stride());
        auto stride_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(stride.data(), stride.size() * sizeof(int32_t), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(stride_buff.get(), 0, 3);

        // src and dst buffers
        auto in_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(src->get_buff_ptr(), src->get_buff_nbytes(), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(in_buff.get(), 0, 4);
        auto out_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(dst->get_buff_ptr(), dst->get_buff_nbytes(), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(out_buff.get(), 0, 5);

        // Dispatch
        auto name = "strided_copy_" + src->get_dtype().str();
        ss_dispatch(ctx, cmd_buff, encoder, name, src->get_numel());
    }
}