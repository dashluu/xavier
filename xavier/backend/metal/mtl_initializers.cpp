#include "mtl_initializers.h"

namespace xv::backend::metal
{
    void full(std::shared_ptr<Array> arr, float c, MTLContext &ctx)
    {
        auto cmd_queue = ctx.get_cmd_queue();
        auto cmd_buff = cmd_queue->commandBuffer();
        auto encoder = cmd_buff->computeCommandEncoder();
        auto device = ctx.get_device();
        auto c_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(&c, sizeof(c), MTL::ResourceStorageModeShared, nullptr));
        auto out_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(arr->get_buff_ptr(), arr->get_buff_nbytes(), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(c_buff.get(), 0, 0);
        encoder->setBuffer(out_buff.get(), 0, 1);
        auto name = "full_" + arr->get_dtype().str();
        ss_dispatch(ctx, cmd_buff, encoder, name, arr->get_numel());
    }

    void arange(std::shared_ptr<Array> arr, int start, int step, MTLContext &ctx)
    {
        auto cmd_queue = ctx.get_cmd_queue();
        auto cmd_buff = cmd_queue->commandBuffer();
        auto encoder = cmd_buff->computeCommandEncoder();
        auto device = ctx.get_device();
        auto start_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(&start, sizeof(start), MTL::ResourceStorageModeShared, nullptr));
        auto step_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(&step, sizeof(step), MTL::ResourceStorageModeShared, nullptr));
        auto out_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(arr->get_buff_ptr(), arr->get_buff_nbytes(), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(start_buff.get(), 0, 0);
        encoder->setBuffer(step_buff.get(), 0, 1);
        encoder->setBuffer(out_buff.get(), 0, 2);
        auto name = "arange_" + arr->get_dtype().str();
        ss_dispatch(ctx, cmd_buff, encoder, name, arr->get_numel());
    }
}