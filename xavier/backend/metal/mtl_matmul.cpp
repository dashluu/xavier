#include "mtl_matmul.h"

void xv::backend::metal::matmul(std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs, std::shared_ptr<Array> output, MTLContext &ctx)
{
    auto cmd_queue = ctx.get_cmd_queue();
    auto cmd_buff = cmd_queue->commandBuffer();
    auto encoder = cmd_buff->computeCommandEncoder();
    auto device = ctx.get_device();

    // Offset
    uint32_t offset[] = {static_cast<uint32_t>(lhs->get_offset()),
                         static_cast<uint32_t>(rhs->get_offset()),
                         static_cast<uint32_t>(output->get_offset())};
    auto offset_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(offset, sizeof(offset), MTL::ResourceStorageModeShared, nullptr));
    encoder->setBuffer(offset_buff.get(), 0, 0);

    // lhs, rhs view
    auto lhs_view = lhs->get_view();
    auto rhs_view = rhs->get_view();
    const uint32_t B = lhs_view[0]; // Batch size
    const uint32_t M = lhs_view[1]; // Number of rows
    const uint32_t K = lhs_view[2]; // Inner dimension
    const uint32_t N = rhs_view[2]; // Number of columns
    uint32_t lhs_view32[] = {B, M, K};
    auto lhs_view_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(lhs_view32, sizeof(lhs_view32), MTL::ResourceStorageModeShared, nullptr));
    encoder->setBuffer(lhs_view_buff.get(), 0, 1);
    uint32_t rhs_view32[] = {B, K, N};
    auto rhs_view_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(rhs_view32, sizeof(rhs_view32), MTL::ResourceStorageModeShared, nullptr));
    encoder->setBuffer(rhs_view_buff.get(), 0, 2);

    // lhs, rhs, output buffers
    auto lhs_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(lhs->get_buff_ptr(), lhs->get_buff_nbytes(), MTL::ResourceStorageModeShared, nullptr));
    encoder->setBuffer(lhs_buff.get(), 0, 3);
    auto rhs_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(rhs->get_buff_ptr(), rhs->get_buff_nbytes(), MTL::ResourceStorageModeShared, nullptr));
    encoder->setBuffer(rhs_buff.get(), 0, 4);
    auto out_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(output->get_buff_ptr(), output->get_buff_nbytes(), MTL::ResourceStorageModeShared, nullptr));
    encoder->setBuffer(out_buff.get(), 0, 5);
    auto kernel_name = "matmul_" + lhs->get_dtype().str();
    auto kernel = ctx.get_kernel(kernel_name);
    encoder->setComputePipelineState(kernel->get_state().get());

    // Compute the number of thread groups
    // Even if matrix is smaller than one threadgroup, we still need at least 1 group
    uint64_t x_group_count = std::max(1ull, static_cast<uint64_t>((N + X_THREADS_PER_GROUP - 1) / X_THREADS_PER_GROUP));
    uint64_t y_group_count = std::max(1ull, static_cast<uint64_t>((M + Y_THREADS_PER_GROUP - 1) / Y_THREADS_PER_GROUP));
    uint64_t z_group_count = std::max(1ull, static_cast<uint64_t>((B + Z_THREADS_PER_GROUP - 1) / Z_THREADS_PER_GROUP));
    auto threadgroup_count = MTL::Size::Make(x_group_count, y_group_count, z_group_count);

    // Compute the number of threads per group
    auto threadgroup_size = MTL::Size::Make(X_THREADS_PER_GROUP, Y_THREADS_PER_GROUP, Z_THREADS_PER_GROUP);
    encoder->dispatchThreadgroups(threadgroup_count, threadgroup_size);
    encoder->endEncoding();
    cmd_buff->commit();
    cmd_buff->waitUntilCompleted();
}

void xv::backend::metal::strided_matmul(std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs, std::shared_ptr<Array> output, MTLContext &ctx)
{
    auto cmd_queue = ctx.get_cmd_queue();
    auto cmd_buff = cmd_queue->commandBuffer();
    auto encoder = cmd_buff->computeCommandEncoder();
    auto device = ctx.get_device();
    // Shared dimensions
    auto ndim = static_cast<uint32_t>(lhs->get_ndim());
    auto ndim_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(&ndim, sizeof(uint32_t), MTL::ResourceStorageModeShared, nullptr));
    encoder->setBuffer(ndim_buff.get(), 0, 0);

    // Offset
    uint32_t offset[] = {static_cast<uint32_t>(lhs->get_offset()),
                         static_cast<uint32_t>(rhs->get_offset()),
                         static_cast<uint32_t>(output->get_offset())};
    auto offset_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(offset, sizeof(offset), MTL::ResourceStorageModeShared, nullptr));
    encoder->setBuffer(offset_buff.get(), 0, 1);

    // lhs, rhs view
    auto lhs_view = lhs->get_view();
    auto rhs_view = rhs->get_view();
    const uint32_t B = lhs_view[0]; // Batch size
    const uint32_t M = lhs_view[1]; // Number of rows
    const uint32_t K = lhs_view[2]; // Inner dimension
    const uint32_t N = rhs_view[2]; // Number of columns
    uint32_t lhs_view32[] = {B, M, K};
    auto lhs_view_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(lhs_view32, sizeof(lhs_view32), MTL::ResourceStorageModeShared, nullptr));
    encoder->setBuffer(lhs_view_buff.get(), 0, 2);
    uint32_t rhs_view32[] = {B, K, N};
    auto rhs_view_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(rhs_view32, sizeof(rhs_view32), MTL::ResourceStorageModeShared, nullptr));
    encoder->setBuffer(rhs_view_buff.get(), 0, 3);

    // lhs, rhs stride
    std::vector<int32_t> lhs_stride = v64to32<int64_t, int32_t>(lhs->get_stride());
    auto lhs_stride_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(lhs_stride.data(), lhs_stride.size() * sizeof(int32_t), MTL::ResourceStorageModeShared, nullptr));
    encoder->setBuffer(lhs_stride_buff.get(), 0, 4);
    std::vector<int32_t> rhs_stride = v64to32<int64_t, int32_t>(rhs->get_stride());
    auto rhs_stride_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(rhs_stride.data(), rhs_stride.size() * sizeof(int32_t), MTL::ResourceStorageModeShared, nullptr));
    encoder->setBuffer(rhs_stride_buff.get(), 0, 5);

    // lhs, rhs, output buffers
    auto lhs_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(lhs->get_buff_ptr(), lhs->get_buff_nbytes(), MTL::ResourceStorageModeShared, nullptr));
    encoder->setBuffer(lhs_buff.get(), 0, 6);
    auto rhs_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(rhs->get_buff_ptr(), rhs->get_buff_nbytes(), MTL::ResourceStorageModeShared, nullptr));
    encoder->setBuffer(rhs_buff.get(), 0, 7);
    auto out_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(output->get_buff_ptr(), output->get_buff_nbytes(), MTL::ResourceStorageModeShared, nullptr));
    encoder->setBuffer(out_buff.get(), 0, 8);
    auto kernel_name = "strided_matmul_" + lhs->get_dtype().str();
    auto kernel = ctx.get_kernel(kernel_name);
    encoder->setComputePipelineState(kernel->get_state().get());

    // Compute the number of thread groups
    // Even if matrix is smaller than one threadgroup, we still need at least 1 group
    uint64_t x_group_count = std::max(1ull, static_cast<uint64_t>((N + X_THREADS_PER_GROUP - 1) / X_THREADS_PER_GROUP));
    uint64_t y_group_count = std::max(1ull, static_cast<uint64_t>((M + Y_THREADS_PER_GROUP - 1) / Y_THREADS_PER_GROUP));
    uint64_t z_group_count = std::max(1ull, static_cast<uint64_t>((B + Z_THREADS_PER_GROUP - 1) / Z_THREADS_PER_GROUP));
    auto threadgroup_count = MTL::Size::Make(x_group_count, y_group_count, z_group_count);

    // Compute the number of threads per group
    auto threadgroup_size = MTL::Size::Make(X_THREADS_PER_GROUP, Y_THREADS_PER_GROUP, Z_THREADS_PER_GROUP);
    encoder->dispatchThreadgroups(threadgroup_count, threadgroup_size);
    encoder->endEncoding();
    cmd_buff->commit();
    cmd_buff->waitUntilCompleted();
}
