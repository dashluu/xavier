#include "mtl_binary.h"

namespace xv::backend::metal
{
    void binary_ss(const std::string &name, ArrayPtr lhs, ArrayPtr rhs, ArrayPtr output, MTLContext &ctx)
    {
        NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
        auto cmd_queue = ctx.get_cmd_queue();
        auto cmd_buff = cmd_queue->commandBuffer();
        auto encoder = cmd_buff->computeCommandEncoder();
        auto device = ctx.get_device();
        uint32_t buff_idx = 0;
        bool strided_input = !lhs->is_contiguous() || !rhs->is_contiguous();
        bool strided_output = !output->is_contiguous();

        // Encode # dimensions if strided input or strided output
        uint32_t ndim;
        if (strided_input || strided_output)
        {
            ndim = static_cast<uint32_t>(lhs->get_ndim());
            encode_buffer(device, encoder, &ndim, sizeof(uint32_t), buff_idx);
        }

        // Encode offset
        encode_offset(device, encoder, {lhs, rhs, output}, buff_idx);

        // Encode view if strided input or strided output
        if (strided_input || strided_output)
        {
            encode_view(device, encoder, lhs, buff_idx);
        }

        // Encode lhs and rhs stride if strided input
        if (strided_input)
        {
            encode_stride(device, encoder, lhs, buff_idx);
            encode_stride(device, encoder, rhs, buff_idx);
        }

        // Encode output stride if strided output
        if (strided_output)
        {
            encode_stride(device, encoder, output, buff_idx);
        }

        // Encode lhs, rhs, and output buffers
        encode_array(device, encoder, lhs, buff_idx);
        encode_array(device, encoder, rhs, buff_idx);
        encode_array(device, encoder, output, buff_idx);

        // Dispatch kernel
        std::string mode = {strided_output ? "s" : "v", strided_input ? "s" : "v"};
        auto kernel_name = name + "_" + mode + "_" + lhs->get_dtype().str();
        ss_dispatch(ctx, cmd_buff, encoder, kernel_name, lhs->get_numel());
        pool->release();
    }
}