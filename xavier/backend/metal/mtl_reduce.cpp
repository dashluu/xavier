#include "mtl_reduce.h"

namespace xv::backend::metal
{
    void reduce_all(const std::string &name, ArrayPtr input, ArrayPtr output, MTLContext &ctx)
    {
        NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
        auto cmd_queue = ctx.get_cmd_queue();
        auto cmd_buff = cmd_queue->commandBuffer();
        auto encoder = cmd_buff->computeCommandEncoder();
        auto device = ctx.get_device();
        uint32_t buff_idx = 0;
        bool strided_input = !input->is_contiguous();

        // Encode # dimensions if strided input
        uint32_t ndim;
        if (strided_input)
        {
            ndim = static_cast<uint32_t>(input->get_ndim());
            encode_buffer(device, encoder, &ndim, sizeof(uint32_t), buff_idx);
        }

        // Encode offset
        encode_offset(device, encoder, {input, output}, buff_idx);

        // Encode input view and stride if strided input
        if (strided_input)
        {
            encode_view(device, encoder, input, buff_idx);
            encode_stride(device, encoder, input, buff_idx);
        }

        // Encode input and output buffers
        encode_array(device, encoder, input, buff_idx);
        encode_array(device, encoder, output, buff_idx);

        // Calculate sizes
        // Get kernel
        auto dtype = input->get_dtype();
        std::string mode = {"v", strided_input ? "s" : "v"};
        auto kernel_name = name + "_all_" + mode + "_" + dtype.str();
        auto kernel = ctx.get_kernel(kernel_name);
        encoder->setComputePipelineState(kernel->get_state().get());
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

        // Dispatch kernel
        encoder->dispatchThreads(threads_per_grid, threads_per_threadgroup);
        encoder->endEncoding();
        cmd_buff->commit();
        cmd_buff->waitUntilCompleted();
        pool->release();
    }

    void reduce_col(const std::string &name, ArrayPtr input, ArrayPtr output, MTLContext &ctx)
    {
        auto cmd_queue = ctx.get_cmd_queue();
        auto cmd_buff = cmd_queue->commandBuffer();
        auto encoder = cmd_buff->computeCommandEncoder();
        auto device = ctx.get_device();
        uint32_t buff_idx = 0;

        // Encode offset
        encode_offset(device, encoder, {input, output}, buff_idx);

        // Encode input view
        auto view = input->get_view();
        const uint32_t M = view[0]; // Number of rows
        const uint32_t N = view[1]; // Number of columns
        uint32_t view32[] = {M, N};
        encode_buffer(device, encoder, view32, sizeof(view32), buff_idx);

        // Encode input and output buffers
        encode_array(device, encoder, input, buff_idx);
        encode_array(device, encoder, output, buff_idx);

        // Calculate sizes
        // Get kernel
        auto dtype = input->get_dtype();
        // TODO: handle the case 'vs'
        std::string mode = "vv";
        auto kernel_name = name + "_col_" + mode + "_" + dtype.str();
        auto kernel = ctx.get_kernel(kernel_name);
        encoder->setComputePipelineState(kernel->get_state().get());
        // This threadgroup_size is applied to each row
        uint32_t max_threads = kernel->get_state()->maxTotalThreadsPerThreadgroup();
        uint32_t simd_size = kernel->get_state()->threadExecutionWidth();
        uint32_t row_threads = std::min(M, max_threads / simd_size);
        uint32_t col_threads = max_threads / row_threads;
        // Ensure threadgroup size is multiple of SIMD size
        col_threads = (col_threads / simd_size) * simd_size;
        // Set threadgroup memory size
        uint32_t threadgroup_nbytes = col_threads * row_threads * dtype.get_size();
        encoder->setThreadgroupMemoryLength(threadgroup_nbytes, 0);
        // Calculate grid size
        MTL::Size threads_per_grid = MTL::Size::Make(N, M, 1);
        MTL::Size threads_per_threadgroup = MTL::Size::Make(col_threads, row_threads, 1);

        // Dispatch kernel
        encoder->dispatchThreads(threads_per_grid, threads_per_threadgroup);
        encoder->endEncoding();
        cmd_buff->commit();
        cmd_buff->waitUntilCompleted();
    }
}