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
    // lhs, rhs view
    auto lhs_view = lhs->get_shape().get_view();
    auto rhs_view = rhs->get_shape().get_view();
    const uint32_t M = lhs_view[0]; // Number of rows
    const uint32_t K = lhs_view[1]; // Inner dimension
    const uint32_t N = rhs_view[1]; // Number of columns
    uint32_t lhs_view32[] = {M, K};
    auto lhs_view_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(lhs_view32, sizeof(lhs_view32), MTL::ResourceStorageModeShared, nullptr));
    encoder->setBuffer(lhs_view_buff.get(), 0, 1);
    uint32_t rhs_view32[] = {K, N};
    auto rhs_view_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(rhs_view32, sizeof(rhs_view32), MTL::ResourceStorageModeShared, nullptr));
    encoder->setBuffer(rhs_view_buff.get(), 0, 2);
    // lhs, rhs, output buffers
    auto lhs_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(lhs->get_buff_ptr(), lhs->get_buff_nbytes(), MTL::ResourceStorageModeShared, nullptr));
    encoder->setBuffer(lhs_buff.get(), 0, 3);
    auto rhs_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(rhs->get_buff_ptr(), rhs->get_buff_nbytes(), MTL::ResourceStorageModeShared, nullptr));
    encoder->setBuffer(rhs_buff.get(), 0, 4);
    auto out_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(output->get_buff_ptr(), output->get_buff_nbytes(), MTL::ResourceStorageModeShared, nullptr));
    encoder->setBuffer(out_buff.get(), 0, 5);
    auto kernel_name = "matmul2d_" + lhs->get_dtype().str();
    auto kernel = ctx.get_kernel(kernel_name);
    encoder->setComputePipelineState(kernel->get_state().get());
    // Compute the number of thread groups
    // Even if matrix is smaller than one threadgroup, we still need at least 1 group
    uint64_t x_group_count = std::max(1ull, (rhs_view[1] + X_THREADS_PER_GROUP - 1) / X_THREADS_PER_GROUP);
    uint64_t y_group_count = std::max(1ull, (lhs_view[0] + Y_THREADS_PER_GROUP - 1) / Y_THREADS_PER_GROUP);
    auto thread_group_count = MTL::Size::Make(x_group_count, y_group_count, 1);
    // Compute the number of threads per group
    auto thread_group_size = MTL::Size::Make(X_THREADS_PER_GROUP, Y_THREADS_PER_GROUP, 1);
    encoder->dispatchThreadgroups(thread_group_count, thread_group_size);
    encoder->endEncoding();
    cmd_buff->commit();
    cmd_buff->waitUntilCompleted();
}