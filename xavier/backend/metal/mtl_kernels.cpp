#include "mtl_kernels.h"

namespace xv::backend::metal
{
    void constant(std::shared_ptr<Array> arr, float c, MTLContext &ctx)
    {
        auto cmd_queue = ctx.get_cmd_queue();
        auto cmd_buff = cmd_queue->commandBuffer();
        auto encoder = cmd_buff->computeCommandEncoder();
        auto name = "constant_c_" + arr->get_dtype().str();
        auto kernel = ctx.get_kernel(name);
        auto device = ctx.get_device();
        auto c_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(&c, sizeof(c), MTL::ResourceStorageModeShared, nullptr));
        auto out_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(arr->get_ptr(), arr->get_nbytes(), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(c_buff.get(), 0, 0);
        encoder->setBuffer(out_buff.get(), 0, 1);
        encoder->setComputePipelineState(kernel->get_state().get());
        auto grid_size = MTL::Size::Make(arr->get_numel(), 1, 1);
        auto thread_group_size = MTL::Size::Make(kernel->get_state()->maxTotalThreadsPerThreadgroup(), 1, 1);
        encoder->dispatchThreads(grid_size, thread_group_size);
        encoder->endEncoding();
        cmd_buff->commit();
        cmd_buff->waitUntilCompleted();
    }

    void arange(std::shared_ptr<Array> arr, int start, int step, MTLContext &ctx)
    {
        auto cmd_queue = ctx.get_cmd_queue();
        auto cmd_buff = cmd_queue->commandBuffer();
        auto encoder = cmd_buff->computeCommandEncoder();
        auto name = "arange_" + arr->get_dtype().str();
        auto kernel = ctx.get_kernel(name);
        auto device = ctx.get_device();
        auto start_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(&start, sizeof(start), MTL::ResourceStorageModeShared, nullptr));
        auto step_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(&step, sizeof(step), MTL::ResourceStorageModeShared, nullptr));
        auto out_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(arr->get_ptr(), arr->get_nbytes(), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(start_buff.get(), 0, 0);
        encoder->setBuffer(step_buff.get(), 0, 1);
        encoder->setBuffer(out_buff.get(), 0, 2);
        encoder->setComputePipelineState(kernel->get_state().get());
        auto grid_size = MTL::Size::Make(arr->get_numel(), 1, 1);
        auto thread_group_size = MTL::Size::Make(kernel->get_state()->maxTotalThreadsPerThreadgroup(), 1, 1);
        encoder->dispatchThreads(grid_size, thread_group_size);
        encoder->endEncoding();
        cmd_buff->commit();
        cmd_buff->waitUntilCompleted();
    }

    void copy(std::shared_ptr<Array> input, std::shared_ptr<Array> output, MTLContext &ctx)
    {
        auto cmd_queue = ctx.get_cmd_queue();
        auto cmd_buff = cmd_queue->commandBuffer();
        auto encoder = cmd_buff->computeCommandEncoder();
        auto name = "copy_" + input->get_dtype().str();
        auto kernel = ctx.get_kernel(name);
        auto device = ctx.get_device();
        auto in_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(input->get_ptr(), input->get_nbytes(), MTL::ResourceStorageModeShared, nullptr));
        auto out_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(output->get_ptr(), output->get_nbytes(), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(in_buff.get(), 0, 0);
        encoder->setBuffer(out_buff.get(), 0, 1);
        encoder->setComputePipelineState(kernel->get_state().get());
        auto grid_size = MTL::Size::Make(input->get_numel(), 1, 1);
        auto thread_group_size = MTL::Size::Make(kernel->get_state()->maxTotalThreadsPerThreadgroup(), 1, 1);
        encoder->dispatchThreads(grid_size, thread_group_size);
        encoder->endEncoding();
        cmd_buff->commit();
        cmd_buff->waitUntilCompleted();
    }

    void sparse_copy(std::shared_ptr<Array> input, std::shared_ptr<Array> output, MTLContext &ctx)
    {
        auto cmd_queue = ctx.get_cmd_queue();
        auto cmd_buff = cmd_queue->commandBuffer();
        auto encoder = cmd_buff->computeCommandEncoder();
        auto name = "sparse_copy_" + input->get_dtype().str();
        auto kernel = ctx.get_kernel(name);
        auto device = ctx.get_device();
        // Input # dimensions
        auto ndim = input->get_ndim();
        auto dim_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(&ndim, sizeof(uint64_t), MTL::ResourceStorageModeShared, nullptr));
        // Input's shape view and stride
        auto &view = input->get_shape().get_view();
        auto &stride = input->get_shape().get_stride();
        auto view_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(view.data(), view.size() * sizeof(uint64_t), MTL::ResourceStorageModeShared, nullptr));
        auto stride_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(stride.data(), stride.size() * sizeof(uint64_t), MTL::ResourceStorageModeShared, nullptr));
        // Input and output
        auto in_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(input->get_ptr(), input->get_nbytes(), MTL::ResourceStorageModeShared, nullptr));
        auto out_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(output->get_ptr(), output->get_nbytes(), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(dim_buff.get(), 0, 0);
        encoder->setBuffer(view_buff.get(), 0, 1);
        encoder->setBuffer(stride_buff.get(), 0, 2);
        encoder->setBuffer(in_buff.get(), 0, 3);
        encoder->setBuffer(out_buff.get(), 0, 4);
        encoder->setComputePipelineState(kernel->get_state().get());
        auto grid_size = MTL::Size::Make(input->get_numel(), 1, 1);
        auto thread_group_size = MTL::Size::Make(kernel->get_state()->maxTotalThreadsPerThreadgroup(), 1, 1);
        encoder->dispatchThreads(grid_size, thread_group_size);
        encoder->endEncoding();
        cmd_buff->commit();
        cmd_buff->waitUntilCompleted();
    }

    void ss_op(const std::string &name, std::vector<std::shared_ptr<Array>> input, std::shared_ptr<Array> output, MTLContext &ctx)
    {
        auto cmd_queue = ctx.get_cmd_queue();
        auto cmd_buff = cmd_queue->commandBuffer();
        auto encoder = cmd_buff->computeCommandEncoder();
        auto kernel_name = name + "_" + output->get_dtype().str();
        auto kernel = ctx.get_kernel(kernel_name);
        auto device = ctx.get_device();
        auto buff_idx = 0;
        for (auto &arr : input)
        {
            auto in_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(arr->get_ptr(), arr->get_nbytes(), MTL::ResourceStorageModeShared, nullptr));
            encoder->setBuffer(in_buff.get(), 0, buff_idx);
            buff_idx++;
        }
        auto out_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(output->get_ptr(), output->get_nbytes(), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(out_buff.get(), 0, buff_idx);
        encoder->setComputePipelineState(kernel->get_state().get());
        auto grid_size = MTL::Size::Make(output->get_numel(), 1, 1);
        auto thread_group_size = MTL::Size::Make(kernel->get_state()->maxTotalThreadsPerThreadgroup(), 1, 1);
        encoder->dispatchThreads(grid_size, thread_group_size);
        encoder->endEncoding();
        cmd_buff->commit();
        cmd_buff->waitUntilCompleted();
    }

    void sparse_ss_op(const std::string &name, std::vector<std::shared_ptr<Array>> input, std::shared_ptr<Array> output, MTLContext &ctx)
    {
        auto cmd_queue = ctx.get_cmd_queue();
        auto cmd_buff = cmd_queue->commandBuffer();
        auto encoder = cmd_buff->computeCommandEncoder();
        auto kernel_name = name + "_" + output->get_dtype().str();
        auto kernel = ctx.get_kernel(kernel_name);
        auto device = ctx.get_device();
        auto buff_idx = 0;
        // Input # dimensions
        auto ndim = output->get_ndim();
        auto dim_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(&ndim, sizeof(uint64_t), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(dim_buff.get(), 0, buff_idx);
        buff_idx++;
        // Input's shape view and stride
        for (auto &arr : input)
        {
            auto &view = arr->get_shape().get_view();
            auto &stride = arr->get_shape().get_stride();
            auto view_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(view.data(), view.size() * sizeof(uint64_t), MTL::ResourceStorageModeShared, nullptr));
            auto stride_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(stride.data(), stride.size() * sizeof(uint64_t), MTL::ResourceStorageModeShared, nullptr));
            encoder->setBuffer(view_buff.get(), 0, buff_idx);
            buff_idx++;
            encoder->setBuffer(stride_buff.get(), 0, buff_idx);
            buff_idx++;
        }
        // Input and output
        for (auto &arr : input)
        {
            auto in_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(arr->get_ptr(), arr->get_nbytes(), MTL::ResourceStorageModeShared, nullptr));
            encoder->setBuffer(in_buff.get(), 0, buff_idx);
            buff_idx++;
        }
        auto out_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(output->get_ptr(), output->get_nbytes(), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(out_buff.get(), 0, buff_idx);
        encoder->setComputePipelineState(kernel->get_state().get());
        auto grid_size = MTL::Size::Make(output->get_numel(), 1, 1);
        auto thread_group_size = MTL::Size::Make(kernel->get_state()->maxTotalThreadsPerThreadgroup(), 1, 1);
        encoder->dispatchThreads(grid_size, thread_group_size);
        encoder->endEncoding();
        cmd_buff->commit();
        cmd_buff->waitUntilCompleted();
    }
}