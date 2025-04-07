#include "mtl_reduce.h"

namespace xv::backend::metal
{
    void reduce_all(const std::string &name, ArrayPtr input, ArrayPtr output, std::shared_ptr<MTLContext> ctx)
    {
        NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
        CommandEncoder encoder(ctx);
        bool strided_input = !input->is_contiguous();

        // Encode buffers
        if (strided_input)
        {
            encoder.encode_ndim(input);
        }
        encoder.encode_offset({input, output});
        if (strided_input)
        {
            encoder.encode_view(input);
            encoder.encode_stride(input);
        }
        encoder.encode_array(input);
        encoder.encode_array(output);
        auto dtype = input->get_dtype();
        const std::string mode = "v" + std::string(strided_input ? "s" : "v");
        const std::string kernel_name = name + "_all_" + mode + "_" + dtype.str();
        encoder.set_pipeline_state(kernel_name);

        const usize max_threadgroup_size = encoder.get_kernel()->get_state()->maxTotalThreadsPerThreadgroup();
        const usize simd_size = encoder.get_kernel()->get_state()->threadExecutionWidth();
        const usize numel = input->get_numel();
        const usize threadgroup_size = std::min(numel, max_threadgroup_size);
        // Set threadgroup memory size
        const usize threadgroup_nbytes = threadgroup_size * dtype.get_size();
        encoder.get_internal_encoder()->setThreadgroupMemoryLength(threadgroup_nbytes, 0);

        // Dispatch kernel
        encoder.dispatch_threads(numel);
        pool->release();
    }

    void reduce_col(const std::string &name, ArrayPtr input, ArrayPtr output, std::shared_ptr<MTLContext> ctx)
    {
        NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
        CommandEncoder encoder(ctx);
        bool strided_input = !input->is_contiguous();

        // Encode buffers
        if (strided_input)
        {
            encoder.encode_ndim(input);
        }
        encoder.encode_offset({input, output});
        encoder.encode_view(input);
        if (strided_input)
        {
            encoder.encode_stride(input);
        }
        encoder.encode_array(input);
        encoder.encode_array(output);
        auto dtype = input->get_dtype();
        const std::string mode = "v" + std::string(strided_input ? "s" : "v");
        const std::string kernel_name = name + "_col_" + mode + "_" + dtype.str();
        encoder.set_pipeline_state(kernel_name);

        auto &view = input->get_view();
        const usize nrows = view[0];
        const usize ncols = view[1];
        const usize max_threadgroup_size = encoder.get_kernel()->get_state()->maxTotalThreadsPerThreadgroup();
        const usize simd_size = encoder.get_kernel()->get_state()->threadExecutionWidth();
        // Ensure column threadgroup and grid size is multiple of SIMD size
        const usize row_grid_size = nrows;
        const usize col_grid_size = ((ncols + simd_size - 1) / simd_size) * simd_size;
        const usize col_threadgroup_size = std::min(col_grid_size, max_threadgroup_size);
        const usize row_threadgroup_size = std::min(row_grid_size, max_threadgroup_size / col_threadgroup_size);
        // Set threadgroup memory size
        const usize threadgroup_nbytes = col_threadgroup_size * row_threadgroup_size * dtype.get_size();
        encoder.get_internal_encoder()->setThreadgroupMemoryLength(threadgroup_nbytes, 0);
        // Compute grid and threadgroup size
        MTL::Size grid_size = MTL::Size::Make(col_grid_size, row_grid_size, 1);
        MTL::Size threadgroup_size = MTL::Size::Make(col_threadgroup_size, row_threadgroup_size, 1);
        std::cout << ctx->get_device()->maxThreadsPerThreadgroup().width << std::endl;
        std::cout << row_grid_size << " " << col_grid_size << std::endl;
        std::cout << row_threadgroup_size << " " << col_threadgroup_size << std::endl;

        // Dispatch kernel
        encoder.dispatch_threads(grid_size, threadgroup_size);
        pool->release();
    }
}