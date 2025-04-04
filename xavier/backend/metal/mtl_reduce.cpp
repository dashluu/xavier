#include "mtl_reduce.h"

namespace xv::backend::metal
{
    void reduce(const std::string &name, std::shared_ptr<Array> input, std::shared_ptr<Array> output, MTLContext &ctx)
    {
        auto cmd_queue = ctx.get_cmd_queue();
        auto cmd_buff = cmd_queue->commandBuffer();
        auto encoder = cmd_buff->computeCommandEncoder();
        auto device = ctx.get_device();

        // Offset
        uint32_t offset[] = {static_cast<uint32_t>(input->get_offset()), static_cast<uint32_t>(output->get_offset())};
        auto offset_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(offset, sizeof(offset), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(offset_buff.get(), 0, 0);

        // Input and output buffers
        auto in_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(input->get_buff_ptr(), input->get_buff_nbytes(), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(in_buff.get(), 0, 1);
        auto out_buff = NS::TransferPtr<MTL::Buffer>(device->newBuffer(output->get_buff_ptr(), output->get_buff_nbytes(), MTL::ResourceStorageModeShared, nullptr));
        encoder->setBuffer(out_buff.get(), 0, 2);
        auto dtype = input->get_dtype();
        auto kernel_name = name + "_" + dtype.str();
        auto kernel = ctx.get_kernel(kernel_name);
        encoder->setComputePipelineState(kernel->get_state().get());

        // Calculate sizes
        uint32_t threadgroup_size = kernel->get_state()->maxTotalThreadsPerThreadgroup();
        uint32_t simd_size = kernel->get_state()->threadExecutionWidth();
        // Ensure threadgroup size is multiple of SIMD size
        threadgroup_size = (threadgroup_size / simd_size) * simd_size;
        // Set threadgroup memory size
        uint32_t threadgroup_nbytes = threadgroup_size * dtype.get_size();
        encoder->setThreadgroupMemoryLength(threadgroup_nbytes, 0);
        // Calculate grid size
        MTL::Size threads_per_grid = MTL::Size::Make(input->get_numel(), 1, 1);
        MTL::Size threads_per_threadgroup = MTL::Size::Make(threadgroup_size, 1, 1);

        // Dispatch
        encoder->dispatchThreads(threads_per_grid, threads_per_threadgroup);
        encoder->endEncoding();
        cmd_buff->commit();
        cmd_buff->waitUntilCompleted();
    }
}