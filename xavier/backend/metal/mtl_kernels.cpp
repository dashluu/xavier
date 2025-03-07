#include "mtl_kernels.h"

namespace xv::backend::metal
{
    void full(std::shared_ptr<Array> arr, float c, MTLContext &ctx)
    {
        auto cmd_queue = ctx.get_cmd_queue();
        auto cmd_buff = cmd_queue->commandBuffer();
        auto encoder = cmd_buff->computeCommandEncoder();
        auto name = "full_" + arr->get_dtype().str();
        auto kernel = ctx.get_kernel(name);
        auto device = ctx.get_device();
        auto c_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(&c, sizeof(c), MTL::ResourceStorageModeShared, nullptr));
        auto out_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(arr->get_buff_ptr(), arr->get_buff_nbytes(), MTL::ResourceStorageModeShared, nullptr));
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
        auto out_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(arr->get_buff_ptr(), arr->get_buff_nbytes(), MTL::ResourceStorageModeShared, nullptr));
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
        auto in_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(input->get_buff_ptr(), input->get_buff_nbytes(), MTL::ResourceStorageModeShared, nullptr));
        auto out_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(output->get_buff_ptr(), output->get_buff_nbytes(), MTL::ResourceStorageModeShared, nullptr));
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
        // # dimensions
        // Cast to 32-bit since the kernel accepts uint buffer
        auto ndim = static_cast<uint32_t>(input->get_ndim());
        auto ndim_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(&ndim, sizeof(uint32_t), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(ndim_buff.get(), 0, 0);
        // Offset
        auto offset = static_cast<uint32_t>(input->get_shape().get_offset());
        auto offset_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(&offset, sizeof(uint32_t), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(offset_buff.get(), 0, 1);
        // View
        auto &view = input->get_shape().get_view();
        // Cast to 32-bit since the kernel accepts uint buffertr
        std::vector<uint32_t> view32(view.size());
        std::transform(view.begin(), view.end(), view32.begin(), [](uint64_t x)
                       { return static_cast<uint32_t>(x); });
        auto view_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(view32.data(), view32.size() * sizeof(uint32_t), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(view_buff.get(), 0, 2);
        // Stride
        auto &stride = input->get_shape().get_stride();
        // Cast to 32-bit since the kernel accepts uint buffer
        std::vector<uint32_t> stride32(stride.size());
        std::transform(stride.begin(), stride.end(), stride32.begin(), [](uint64_t x)
                       { return static_cast<uint32_t>(x); });
        auto stride_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(stride32.data(), stride32.size() * sizeof(uint32_t), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(stride_buff.get(), 0, 3);
        // Data
        auto in_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(input->get_buff_ptr(), input->get_buff_nbytes(), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(in_buff.get(), 0, 4);
        auto out_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(output->get_buff_ptr(), output->get_buff_nbytes(), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(out_buff.get(), 0, 5);
        encoder->setComputePipelineState(kernel->get_state().get());
        auto grid_size = MTL::Size::Make(input->get_numel(), 1, 1);
        auto thread_group_size = MTL::Size::Make(kernel->get_state()->maxTotalThreadsPerThreadgroup(), 1, 1);
        encoder->dispatchThreads(grid_size, thread_group_size);
        encoder->endEncoding();
        cmd_buff->commit();
        cmd_buff->waitUntilCompleted();
    }

    void ss_op(const std::string &name, std::vector<std::shared_ptr<Array>> input, MTLContext &ctx)
    {
        auto cmd_queue = ctx.get_cmd_queue();
        auto cmd_buff = cmd_queue->commandBuffer();
        auto encoder = cmd_buff->computeCommandEncoder();
        auto kernel_name = name + "_" + input[0]->get_dtype().str();
        auto kernel = ctx.get_kernel(kernel_name);
        auto device = ctx.get_device();
        auto buff_idx = 0;
        for (auto &arr : input)
        {
            auto in_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(arr->get_buff_ptr(), arr->get_buff_nbytes(), MTL::ResourceStorageModeShared, nullptr));
            encoder->setBuffer(in_buff.get(), 0, buff_idx);
            buff_idx++;
        }
        encoder->setComputePipelineState(kernel->get_state().get());
        auto grid_size = MTL::Size::Make(input[0]->get_numel(), 1, 1);
        auto thread_group_size = MTL::Size::Make(kernel->get_state()->maxTotalThreadsPerThreadgroup(), 1, 1);
        encoder->dispatchThreads(grid_size, thread_group_size);
        encoder->endEncoding();
        cmd_buff->commit();
        cmd_buff->waitUntilCompleted();
    }

    void sparse_ss_op(const std::string &name, std::vector<std::shared_ptr<Array>> input, MTLContext &ctx)
    {
        auto cmd_queue = ctx.get_cmd_queue();
        auto cmd_buff = cmd_queue->commandBuffer();
        auto encoder = cmd_buff->computeCommandEncoder();
        auto kernel_name = name + "_" + input[0]->get_dtype().str();
        auto kernel = ctx.get_kernel(kernel_name);
        auto device = ctx.get_device();
        auto buff_idx = 0;
        // # dimensions
        // Cast to 32-bit since the kernel accepts uint buffer
        auto ndim = input[0]->get_ndim();
        auto ndim32 = static_cast<uint32_t>(ndim);
        auto ndim_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(&ndim32, sizeof(uint32_t), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(ndim_buff.get(), 0, buff_idx);
        buff_idx++;
        std::vector<uint32_t> view32(ndim);
        std::vector<uint32_t> stride32(ndim);
        for (auto &arr : input)
        {
            // Offset
            // Cast to 32-bit since the kernel accepts uint buffer
            auto offset = static_cast<uint32_t>(arr->get_shape().get_offset());
            auto offset_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(&offset, sizeof(uint32_t), MTL::ResourceStorageModeShared, nullptr));
            encoder->setBuffer(offset_buff.get(), 0, buff_idx);
            buff_idx++;
            // View
            auto &view = arr->get_shape().get_view();
            // Cast to 32-bit since the kernel accepts uint buffertr
            std::transform(view.begin(), view.end(), view32.begin(), [](uint64_t x)
                           { return static_cast<uint32_t>(x); });
            auto view_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(view32.data(), view32.size() * sizeof(uint32_t), MTL::ResourceStorageModeShared, nullptr));
            encoder->setBuffer(view_buff.get(), 0, buff_idx);
            buff_idx++;
            // Stride
            auto &stride = arr->get_shape().get_stride();
            // Cast to 32-bit since the kernel accepts uint buffer
            std::transform(stride.begin(), stride.end(), stride32.begin(), [](uint64_t x)
                           { return static_cast<uint32_t>(x); });
            auto stride_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(stride32.data(), stride32.size() * sizeof(uint32_t), MTL::ResourceStorageModeShared, nullptr));
            encoder->setBuffer(stride_buff.get(), 0, buff_idx);
            buff_idx++;
        }
        for (auto &arr : input)
        {
            // Data
            auto in_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(arr->get_buff_ptr(), arr->get_buff_nbytes(), MTL::ResourceStorageModeShared, nullptr));
            encoder->setBuffer(in_buff.get(), 0, buff_idx);
            buff_idx++;
        }
        encoder->setComputePipelineState(kernel->get_state().get());
        auto grid_size = MTL::Size::Make(input[0]->get_numel(), 1, 1);
        auto thread_group_size = MTL::Size::Make(kernel->get_state()->maxTotalThreadsPerThreadgroup(), 1, 1);
        encoder->dispatchThreads(grid_size, thread_group_size);
        encoder->endEncoding();
        cmd_buff->commit();
        cmd_buff->waitUntilCompleted();
    }
}