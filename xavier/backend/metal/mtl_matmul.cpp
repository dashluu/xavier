#include "mtl_matmul.h"

void xv::backend::metal::matmul(ArrayPtr lhs, ArrayPtr rhs, ArrayPtr output, MTLContext &ctx)
{
    auto cmd_queue = ctx.get_cmd_queue();
    auto cmd_buff = cmd_queue->commandBuffer();
    auto encoder = cmd_buff->computeCommandEncoder();
    auto device = ctx.get_device();
    uint32_t buff_idx = 0;
    bool strided_input = !lhs->is_contiguous() || !rhs->is_contiguous();

    // Encode # dimensions if strided input
    uint32_t ndim;
    if (strided_input)
    {
        ndim = static_cast<uint32_t>(lhs->get_ndim());
        encode_buffer(device, encoder, &ndim, sizeof(uint32_t), buff_idx);
    }

    // Encode offset
    encode_offset(device, encoder, {lhs, rhs, output}, buff_idx);

    // Encode lhs, rhs view
    auto lhs_view = lhs->get_view();
    auto rhs_view = rhs->get_view();
    const uint32_t B = lhs_view[0]; // Batch size
    const uint32_t M = lhs_view[1]; // Number of rows
    const uint32_t K = lhs_view[2]; // Inner dimension
    const uint32_t N = rhs_view[2]; // Number of columns
    uint32_t lhs_view32[] = {B, M, K};
    encode_buffer(device, encoder, lhs_view32, sizeof(lhs_view32), buff_idx);
    uint32_t rhs_view32[] = {B, K, N};
    encode_buffer(device, encoder, rhs_view32, sizeof(rhs_view32), buff_idx);

    // Encode lhs and rhs stride if strided input
    if (strided_input)
    {
        encode_stride(device, encoder, lhs, buff_idx);
        encode_stride(device, encoder, rhs, buff_idx);
    }

    // Encode lhs, rhs, and output buffers
    encode_array(device, encoder, lhs, buff_idx);
    encode_array(device, encoder, rhs, buff_idx);
    encode_array(device, encoder, output, buff_idx);

    // Calculate sizes
    // Compute the number of thread groups
    // Even if matrix is smaller than one threadgroup, we still need at least 1 group
    uint64_t x_group_count = std::max(1ull, static_cast<uint64_t>((N + X_THREADS_PER_GROUP - 1) / X_THREADS_PER_GROUP));
    uint64_t y_group_count = std::max(1ull, static_cast<uint64_t>((M + Y_THREADS_PER_GROUP - 1) / Y_THREADS_PER_GROUP));
    uint64_t z_group_count = std::max(1ull, static_cast<uint64_t>((B + Z_THREADS_PER_GROUP - 1) / Z_THREADS_PER_GROUP));
    auto threadgroup_count = MTL::Size::Make(x_group_count, y_group_count, z_group_count);
    // Compute the number of threads per group
    auto threadgroup_size = MTL::Size::Make(X_THREADS_PER_GROUP, Y_THREADS_PER_GROUP, Z_THREADS_PER_GROUP);

    // Dispatch kernel
    std::string mode = {"v", strided_input ? "s" : "v"};
    auto kernel_name = "matmul_" + mode + "_" + lhs->get_dtype().str();
    auto kernel = ctx.get_kernel(kernel_name);
    encoder->setComputePipelineState(kernel->get_state().get());
    encoder->dispatchThreadgroups(threadgroup_count, threadgroup_size);
    encoder->endEncoding();
    cmd_buff->commit();
    cmd_buff->waitUntilCompleted();
}
