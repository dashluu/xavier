#include "mtl_binary.h"

namespace xv::backend::metal
{
    void binary_ss(const std::string &name, std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs, std::shared_ptr<Array> output, MTLContext &ctx)
    {
        auto cmd_queue = ctx.get_cmd_queue();
        auto cmd_buff = cmd_queue->commandBuffer();
        auto encoder = cmd_buff->computeCommandEncoder();
        auto device = ctx.get_device();
        // Offset
        uint32_t offset[] = {static_cast<uint32_t>(lhs->get_shape().get_offset()), static_cast<uint32_t>(rhs->get_shape().get_offset())};
        auto offset_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(offset, sizeof(offset), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(offset_buff.get(), 0, 0);
        // lhs, rhs, output buffers
        auto lhs_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(lhs->get_buff_ptr(), lhs->get_buff_nbytes(), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(lhs_buff.get(), 0, 1);
        auto rhs_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(rhs->get_buff_ptr(), rhs->get_buff_nbytes(), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(rhs_buff.get(), 0, 2);
        auto out_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(output->get_buff_ptr(), output->get_buff_nbytes(), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(out_buff.get(), 0, 3);
        auto kernel_name = name + "_" + lhs->get_dtype().str();
        ss_dispatch(ctx, cmd_buff, encoder, kernel_name, lhs->get_numel());
    }

    void self_binary_ss(const std::string &name, std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs, MTLContext &ctx)
    {
        auto cmd_queue = ctx.get_cmd_queue();
        auto cmd_buff = cmd_queue->commandBuffer();
        auto encoder = cmd_buff->computeCommandEncoder();
        auto device = ctx.get_device();
        // Offset
        uint32_t offset[] = {static_cast<uint32_t>(lhs->get_shape().get_offset()), static_cast<uint32_t>(rhs->get_shape().get_offset())};
        auto offset_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(offset, sizeof(offset), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(offset_buff.get(), 0, 0);
        // lhs, rhs, output buffers
        auto lhs_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(lhs->get_buff_ptr(), lhs->get_buff_nbytes(), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(lhs_buff.get(), 0, 1);
        auto rhs_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(rhs->get_buff_ptr(), rhs->get_buff_nbytes(), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(rhs_buff.get(), 0, 2);
        auto kernel_name = name + "_" + lhs->get_dtype().str();
        ss_dispatch(ctx, cmd_buff, encoder, kernel_name, lhs->get_numel());
    }

    void sparse_binary_ss(const std::string &name, std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs, std::shared_ptr<Array> output, MTLContext &ctx)
    {
        auto cmd_queue = ctx.get_cmd_queue();
        auto cmd_buff = cmd_queue->commandBuffer();
        auto encoder = cmd_buff->computeCommandEncoder();
        auto device = ctx.get_device();

        // Shared dimensions (since lhs and rhs match)
        auto ndim = static_cast<uint32_t>(lhs->get_ndim());
        auto ndim_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(&ndim, sizeof(uint32_t), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(ndim_buff.get(), 0, 0);

        // Offset
        uint32_t offset[] = {static_cast<uint32_t>(lhs->get_shape().get_offset()), static_cast<uint32_t>(rhs->get_shape().get_offset())};
        auto offset_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(offset, sizeof(offset), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(offset_buff.get(), 0, 1);

        // View (shared since dimensions match)
        auto view = lhs->get_shape().get_view();
        std::vector<uint32_t> view32 = v64to32<uint64_t, uint32_t>(view);
        auto view_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(view32.data(), view32.size() * sizeof(uint32_t), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(view_buff.get(), 0, 2);

        // lhs stride
        auto lhs_stride = lhs->get_shape().get_stride();
        std::vector<int32_t> lhs_stride32 = v64to32<int64_t, int32_t>(lhs_stride);
        auto lhs_stride_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(lhs_stride32.data(), lhs_stride32.size() * sizeof(int32_t), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(lhs_stride_buff.get(), 0, 3);

        // rhs stride
        auto rhs_stride = rhs->get_shape().get_stride();
        std::vector<int32_t> rhs_stride32 = v64to32<int64_t, int32_t>(rhs_stride);
        auto rhs_stride_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(rhs_stride32.data(), rhs_stride32.size() * sizeof(int32_t), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(rhs_stride_buff.get(), 0, 4);

        // lhs, rhs, output buffers
        auto lhs_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(lhs->get_buff_ptr(), lhs->get_buff_nbytes(), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(lhs_buff.get(), 0, 5);
        auto rhs_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(rhs->get_buff_ptr(), rhs->get_buff_nbytes(), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(rhs_buff.get(), 0, 6);
        auto out_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(output->get_buff_ptr(), output->get_buff_nbytes(), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(out_buff.get(), 0, 7);

        // Dispatch
        auto kernel_name = "sparse_" + name + "_" + lhs->get_dtype().str();
        ss_dispatch(ctx, cmd_buff, encoder, kernel_name, lhs->get_numel());
    }

    void sparse_self_binary_ss(const std::string &name, std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs, MTLContext &ctx)
    {
        auto cmd_queue = ctx.get_cmd_queue();
        auto cmd_buff = cmd_queue->commandBuffer();
        auto encoder = cmd_buff->computeCommandEncoder();
        auto device = ctx.get_device();

        // Shared dimensions (since lhs and rhs match)
        auto ndim = static_cast<uint32_t>(lhs->get_ndim());
        auto ndim_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(&ndim, sizeof(uint32_t), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(ndim_buff.get(), 0, 0);

        // Offset
        uint32_t offset[] = {static_cast<uint32_t>(lhs->get_shape().get_offset()), static_cast<uint32_t>(rhs->get_shape().get_offset())};
        auto offset_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(offset, sizeof(offset), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(offset_buff.get(), 0, 1);

        // View (shared since dimensions match)
        auto view = lhs->get_shape().get_view();
        std::vector<uint32_t> view32 = v64to32<uint64_t, uint32_t>(view);
        auto view_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(view32.data(), view32.size() * sizeof(uint32_t), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(view_buff.get(), 0, 2);

        // lhs stride
        auto lhs_stride = lhs->get_shape().get_stride();
        std::vector<int32_t> lhs_stride32 = v64to32<int64_t, int32_t>(lhs_stride);
        auto lhs_stride_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(lhs_stride32.data(), lhs_stride32.size() * sizeof(int32_t), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(lhs_stride_buff.get(), 0, 3);

        // rhs stride
        auto rhs_stride = rhs->get_shape().get_stride();
        std::vector<int32_t> rhs_stride32 = v64to32<int64_t, int32_t>(rhs_stride);
        auto rhs_stride_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(rhs_stride32.data(), rhs_stride32.size() * sizeof(int32_t), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(rhs_stride_buff.get(), 0, 4);

        // lhs, rhs, output buffers
        auto lhs_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(lhs->get_buff_ptr(), lhs->get_buff_nbytes(), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(lhs_buff.get(), 0, 5);
        auto rhs_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(rhs->get_buff_ptr(), rhs->get_buff_nbytes(), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(rhs_buff.get(), 0, 6);

        // Dispatch
        auto kernel_name = "sparse_" + name + "_" + lhs->get_dtype().str();
        ss_dispatch(ctx, cmd_buff, encoder, kernel_name, lhs->get_numel());
    }
}