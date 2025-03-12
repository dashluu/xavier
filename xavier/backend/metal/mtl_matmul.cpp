#include "mtl_matmul.h"

void xv::backend::metal::matmul2d(std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs, std::shared_ptr<Array> output, MTLContext &ctx)
{
    auto cmd_queue = ctx.get_cmd_queue();
    auto cmd_buff = cmd_queue->commandBuffer();
    auto encoder = cmd_buff->computeCommandEncoder();
    auto device = ctx.get_device();
    // Offset
    uint32_t offset[] = {static_cast<uint32_t>(lhs->get_shape().get_offset()), static_cast<uint32_t>(rhs->get_shape().get_offset())};
    auto offset_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(offset, sizeof(offset), MTL::ResourceStorageModeShared, nullptr));
    encoder->setBuffer(offset_buff.get(), 0, 0);
    // lhs view
    auto lhs_view = lhs->get_shape().get_view();
    std::vector<uint32_t> lhs_view32 = v64to32<uint64_t, uint32_t>(lhs_view);
    auto lhs_view_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(lhs_view32.data(), lhs_view32.size() * sizeof(int32_t), MTL::ResourceStorageModeShared, nullptr));
    encoder->setBuffer(lhs_view_buff.get(), 0, 1);
    // rhs view
    auto rhs_view = rhs->get_shape().get_view();
    std::vector<uint32_t> rhs_view32 = v64to32<uint64_t, uint32_t>(rhs_view);
    auto rhs_view_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(rhs_view32.data(), rhs_view32.size() * sizeof(int32_t), MTL::ResourceStorageModeShared, nullptr));
    encoder->setBuffer(rhs_view_buff.get(), 0, 2);
    // lhs, rhs, output buffers
    auto lhs_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(lhs->get_buff_ptr(), lhs->get_buff_nbytes(), MTL::ResourceStorageModeShared, nullptr));
    encoder->setBuffer(lhs_buff.get(), 0, 3);
    auto rhs_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(rhs->get_buff_ptr(), rhs->get_buff_nbytes(), MTL::ResourceStorageModeShared, nullptr));
    encoder->setBuffer(rhs_buff.get(), 0, 4);
    auto out_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(output->get_buff_ptr(), output->get_buff_nbytes(), MTL::ResourceStorageModeShared, nullptr));
    encoder->setBuffer(out_buff.get(), 0, 5);
    auto kernel_name = "matmul2d_" + lhs->get_dtype().str();
    ss_dispatch(ctx, cmd_buff, encoder, kernel_name, lhs->get_numel());
}