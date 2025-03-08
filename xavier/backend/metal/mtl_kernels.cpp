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
        std::vector<uint32_t> view32 = vec64to32<uint64_t, uint32_t>(view);
        auto view_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(view32.data(), view32.size() * sizeof(uint32_t), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(view_buff.get(), 0, 2);

        // Stride
        auto &stride = input->get_shape().get_stride();
        // Cast to 32-bit since the kernel accepts uint buffer
        std::vector<int32_t> stride32 = vec64to32<int64_t, int32_t>(stride);
        auto stride_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(stride32.data(), stride32.size() * sizeof(int32_t), MTL::ResourceStorageModeShared, nullptr));
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

    void unary_ss(const std::string &name, std::shared_ptr<Array> input, std::shared_ptr<Array> output, MTLContext &ctx)
    {
        auto cmd_queue = ctx.get_cmd_queue();
        auto cmd_buff = cmd_queue->commandBuffer();
        auto encoder = cmd_buff->computeCommandEncoder();
        auto kernel_name = name + "_" + input->get_dtype().str();
        auto kernel = ctx.get_kernel(kernel_name);
        auto device = ctx.get_device();
        auto in_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(input->get_buff_ptr(), input->get_buff_nbytes(), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(in_buff.get(), 0, 0);
        auto out_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(output->get_buff_ptr(), output->get_buff_nbytes(), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(out_buff.get(), 0, 1);
        encoder->setComputePipelineState(kernel->get_state().get());
        auto grid_size = MTL::Size::Make(input->get_numel(), 1, 1);
        auto thread_group_size = MTL::Size::Make(kernel->get_state()->maxTotalThreadsPerThreadgroup(), 1, 1);
        encoder->dispatchThreads(grid_size, thread_group_size);
        encoder->endEncoding();
        cmd_buff->commit();
        cmd_buff->waitUntilCompleted();
    }

    void binary_ss(const std::string &name, std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs, std::shared_ptr<Array> output, MTLContext &ctx)
    {
        auto cmd_queue = ctx.get_cmd_queue();
        auto cmd_buff = cmd_queue->commandBuffer();
        auto encoder = cmd_buff->computeCommandEncoder();
        auto kernel_name = name + "_" + lhs->get_dtype().str();
        auto kernel = ctx.get_kernel(kernel_name);
        auto device = ctx.get_device();
        auto lhs_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(lhs->get_buff_ptr(), lhs->get_buff_nbytes(), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(lhs_buff.get(), 0, 0);
        auto rhs_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(rhs->get_buff_ptr(), rhs->get_buff_nbytes(), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(rhs_buff.get(), 0, 1);
        auto out_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(output->get_buff_ptr(), output->get_buff_nbytes(), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(out_buff.get(), 0, 2);
        encoder->setComputePipelineState(kernel->get_state().get());
        auto grid_size = MTL::Size::Make(lhs->get_numel(), 1, 1);
        auto thread_group_size = MTL::Size::Make(kernel->get_state()->maxTotalThreadsPerThreadgroup(), 1, 1);
        encoder->dispatchThreads(grid_size, thread_group_size);
        encoder->endEncoding();
        cmd_buff->commit();
        cmd_buff->waitUntilCompleted();
    }

    void sparse_unary_ss(const std::string &name, std::shared_ptr<Array> input, std::shared_ptr<Array> output, MTLContext &ctx)
    {
        auto cmd_queue = ctx.get_cmd_queue();
        auto cmd_buff = cmd_queue->commandBuffer();
        auto encoder = cmd_buff->computeCommandEncoder();
        auto kernel_name = name + "_" + input->get_dtype().str();
        auto kernel = ctx.get_kernel(kernel_name);
        auto device = ctx.get_device();

        // # dimensions
        auto ndim = static_cast<uint32_t>(input->get_ndim());
        auto ndim_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(&ndim, sizeof(uint32_t), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(ndim_buff.get(), 0, 0);

        // Offset
        auto offset = static_cast<uint32_t>(input->get_shape().get_offset());
        auto offset_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(&offset, sizeof(uint32_t), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(offset_buff.get(), 0, 1);

        // View
        auto view = input->get_shape().get_view();
        std::vector<uint32_t> view32 = vec64to32<uint64_t, uint32_t>(view);
        auto view_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(view32.data(), view32.size() * sizeof(uint32_t), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(view_buff.get(), 0, 2);

        // Stride
        auto stride = input->get_shape().get_stride();
        std::vector<int32_t> stride32 = vec64to32<int64_t, int32_t>(stride);
        auto stride_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(stride32.data(), stride32.size() * sizeof(int32_t), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(stride_buff.get(), 0, 3);

        // Input and output buffers
        auto in_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(input->get_buff_ptr(), input->get_buff_nbytes(), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(in_buff.get(), 0, 4);
        auto out_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(output->get_buff_ptr(), output->get_buff_nbytes(), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(out_buff.get(), 0, 5);

        // Dispatch
        encoder->setComputePipelineState(kernel->get_state().get());
        auto grid_size = MTL::Size::Make(input->get_numel(), 1, 1);
        auto thread_group_size = MTL::Size::Make(kernel->get_state()->maxTotalThreadsPerThreadgroup(), 1, 1);
        encoder->dispatchThreads(grid_size, thread_group_size);
        encoder->endEncoding();
        cmd_buff->commit();
        cmd_buff->waitUntilCompleted();
    }

    void sparse_binary_ss(const std::string &name, std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs, std::shared_ptr<Array> output, MTLContext &ctx)
    {
        auto cmd_queue = ctx.get_cmd_queue();
        auto cmd_buff = cmd_queue->commandBuffer();
        auto encoder = cmd_buff->computeCommandEncoder();
        auto kernel_name = name + "_" + lhs->get_dtype().str();
        auto kernel = ctx.get_kernel(kernel_name);
        auto device = ctx.get_device();

        // Shared dimensions (since lhs and rhs match)
        auto ndim = static_cast<uint32_t>(lhs->get_ndim());
        auto ndim_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(&ndim, sizeof(uint32_t), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(ndim_buff.get(), 0, 0);

        // View (shared since dimensions match)
        auto view = lhs->get_shape().get_view();
        std::vector<uint32_t> view32 = vec64to32<uint64_t, uint32_t>(view);
        auto view_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(view32.data(), view32.size() * sizeof(uint32_t), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(view_buff.get(), 0, 1);

        // lhs offset
        auto lhs_offset = static_cast<uint32_t>(lhs->get_shape().get_offset());
        auto lhs_offset_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(&lhs_offset, sizeof(uint32_t), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(lhs_offset_buff.get(), 0, 2);

        // lhs stride
        auto lhs_stride = lhs->get_shape().get_stride();
        std::vector<int32_t> lhs_stride32 = vec64to32<int64_t, int32_t>(lhs_stride);
        auto lhs_stride_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(lhs_stride32.data(), lhs_stride32.size() * sizeof(int32_t), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(lhs_stride_buff.get(), 0, 3);

        // rhs offset
        auto rhs_offset = static_cast<uint32_t>(rhs->get_shape().get_offset());
        auto rhs_offset_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(&rhs_offset, sizeof(uint32_t), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(rhs_offset_buff.get(), 0, 4);

        // rhs stride
        auto rhs_stride = rhs->get_shape().get_stride();
        std::vector<int32_t> rhs_stride32 = vec64to32<int64_t, int32_t>(rhs_stride);
        auto rhs_stride_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(rhs_stride32.data(), rhs_stride32.size() * sizeof(int32_t), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(rhs_stride_buff.get(), 0, 5);

        // Input and output buffers
        auto lhs_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(lhs->get_buff_ptr(), lhs->get_buff_nbytes(), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(lhs_buff.get(), 0, 6);
        auto rhs_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(rhs->get_buff_ptr(), rhs->get_buff_nbytes(), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(rhs_buff.get(), 0, 7);
        auto out_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(output->get_buff_ptr(), output->get_buff_nbytes(), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(out_buff.get(), 0, 8);

        // Dispatch
        encoder->setComputePipelineState(kernel->get_state().get());
        auto grid_size = MTL::Size::Make(output->get_numel(), 1, 1);
        auto thread_group_size = MTL::Size::Make(kernel->get_state()->maxTotalThreadsPerThreadgroup(), 1, 1);
        encoder->dispatchThreads(grid_size, thread_group_size);
        encoder->endEncoding();
        cmd_buff->commit();
        cmd_buff->waitUntilCompleted();
    }
}