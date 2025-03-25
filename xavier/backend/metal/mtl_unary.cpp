#include "mtl_unary.h"

namespace xv::backend::metal
{
    void unary_ss(const std::string &name, std::shared_ptr<Array> input, std::shared_ptr<Array> output, MTLContext &ctx)
    {
        auto cmd_queue = ctx.get_cmd_queue();
        auto cmd_buff = cmd_queue->commandBuffer();
        auto encoder = cmd_buff->computeCommandEncoder();
        auto device = ctx.get_device();
        // Offset
        uint32_t offset[] = {static_cast<uint32_t>(input->get_shape().get_offset()), static_cast<uint32_t>(output->get_shape().get_offset())};
        auto offset_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(offset, sizeof(offset), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(offset_buff.get(), 0, 0);
        // Input and output buffers
        auto in_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(input->get_buff_ptr(), input->get_buff_nbytes(), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(in_buff.get(), 0, 1);
        auto out_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(output->get_buff_ptr(), output->get_buff_nbytes(), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(out_buff.get(), 0, 2);
        auto kernel_name = name + "_" + input->get_dtype().str();
        ss_dispatch(ctx, cmd_buff, encoder, kernel_name, input->get_numel());
    }

    void sparse_unary_ss(const std::string &name, std::shared_ptr<Array> input, std::shared_ptr<Array> output, MTLContext &ctx)
    {
        auto cmd_queue = ctx.get_cmd_queue();
        auto cmd_buff = cmd_queue->commandBuffer();
        auto encoder = cmd_buff->computeCommandEncoder();
        auto device = ctx.get_device();

        // # dimensions
        auto ndim = static_cast<uint32_t>(input->get_ndim());
        auto ndim_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(&ndim, sizeof(uint32_t), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(ndim_buff.get(), 0, 0);

        // Offset
        uint32_t offset[] = {static_cast<uint32_t>(input->get_shape().get_offset()), static_cast<uint32_t>(output->get_shape().get_offset())};
        auto offset_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(offset, sizeof(offset), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(offset_buff.get(), 0, 1);

        // View
        auto view = input->get_shape().get_view();
        std::vector<uint32_t> view32 = v64to32<uint64_t, uint32_t>(view);
        auto view_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(view32.data(), view32.size() * sizeof(uint32_t), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(view_buff.get(), 0, 2);

        // Stride
        auto stride = input->get_shape().get_stride();
        std::vector<int32_t> stride32 = v64to32<int64_t, int32_t>(stride);
        auto stride_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(stride32.data(), stride32.size() * sizeof(int32_t), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(stride_buff.get(), 0, 3);

        // Input and output buffers
        auto in_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(input->get_buff_ptr(), input->get_buff_nbytes(), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(in_buff.get(), 0, 4);
        auto out_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(output->get_buff_ptr(), output->get_buff_nbytes(), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(out_buff.get(), 0, 5);

        // Dispatch
        auto kernel_name = "sparse_" + name + "_" + input->get_dtype().str();
        ss_dispatch(ctx, cmd_buff, encoder, kernel_name, input->get_numel());
    }
}