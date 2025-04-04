#include "mtl_initializers.h"

namespace xv::backend::metal
{
    void full(ArrayPtr arr, int c, usize size, MTLContext &ctx)
    {
        NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
        auto cmd_queue = ctx.get_cmd_queue();
        auto cmd_buff = cmd_queue->commandBuffer();
        auto encoder = cmd_buff->computeCommandEncoder();
        auto device = ctx.get_device();
        uint32_t buff_idx = 0;
        encode_buffer(device, encoder, &c, size, buff_idx);
        encode_array(device, encoder, arr, buff_idx);
        auto name = "full_" + arr->get_dtype().str();
        ss_dispatch(ctx, cmd_buff, encoder, name, arr->get_numel());
        pool->release();
    }

    void arange(ArrayPtr arr, int start, int step, MTLContext &ctx)
    {
        NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
        auto cmd_queue = ctx.get_cmd_queue();
        auto cmd_buff = cmd_queue->commandBuffer();
        auto encoder = cmd_buff->computeCommandEncoder();
        auto device = ctx.get_device();
        uint32_t buff_idx = 0;
        encode_buffer(device, encoder, &start, sizeof(start), buff_idx);
        encode_buffer(device, encoder, &step, sizeof(step), buff_idx);
        encode_array(device, encoder, arr, buff_idx);
        auto name = "arange_" + arr->get_dtype().str();
        ss_dispatch(ctx, cmd_buff, encoder, name, arr->get_numel());
        pool->release();
    }
}