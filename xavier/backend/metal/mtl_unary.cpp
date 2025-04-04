#include "mtl_unary.h"

namespace xv::backend::metal
{
    void unary_ss(const std::string &name, ArrayPtr input, ArrayPtr output, MTLContext &ctx)
    {
        NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
        auto cmd_queue = ctx.get_cmd_queue();
        auto cmd_buff = cmd_queue->commandBuffer();
        auto encoder = cmd_buff->computeCommandEncoder();
        auto device = ctx.get_device();
        uint32_t buff_idx = 0;
        bool strided_input = !input->is_contiguous();
        bool strided_output = !output->is_contiguous();

        // Encode # dimensions if strided input or strided output
        uint32_t ndim;
        if (strided_input || strided_output)
        {
            ndim = static_cast<uint32_t>(input->get_ndim());
            encode_buffer(device, encoder, &ndim, sizeof(uint32_t), buff_idx);
        }

        // Encode offset
        encode_offset(device, encoder, {input, output}, buff_idx);

        // Encode view if strided input or strided output
        if (strided_input || strided_output)
        {
            encode_view(device, encoder, input, buff_idx);
        }

        // Encode input stride if strided input
        if (strided_input)
        {
            encode_stride(device, encoder, input, buff_idx);
        }

        // Encode output stride if strided output
        if (strided_output)
        {
            encode_stride(device, encoder, output, buff_idx);
        }

        // Encode input and output buffers
        encode_array(device, encoder, input, buff_idx);
        encode_array(device, encoder, output, buff_idx);

        // Dispatch kernel
        std::string mode = {strided_output ? "s" : "v", strided_input ? "s" : "v"};
        auto kernel_name = name + "_" + mode + "_" + input->get_dtype().str();
        ss_dispatch(ctx, cmd_buff, encoder, kernel_name, input->get_numel());
        pool->release();
    }
}