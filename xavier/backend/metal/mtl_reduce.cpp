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

        uint32_t max_threadgroup_size = encoder.get_kernel()->get_state()->maxTotalThreadsPerThreadgroup();
        uint32_t simd_size = encoder.get_kernel()->get_state()->threadExecutionWidth();
        // Ensure threadgroup size is multiple of SIMD size
        max_threadgroup_size = (max_threadgroup_size / simd_size) * simd_size;
        // Set threadgroup memory size
        uint32_t threadgroup_nbytes = max_threadgroup_size * dtype.get_size();
        encoder.get_internal_encoder()->setThreadgroupMemoryLength(threadgroup_nbytes, 0);

        // Dispatch kernel
        encoder.dispatch_threads(input->get_numel());
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
        if (strided_input)
        {
            encoder.encode_view(input);
            encoder.encode_stride(input);
        }
        encoder.encode_array(input);
        encoder.encode_array(output);
        auto dtype = input->get_dtype();
        const std::string mode = "v" + std::string(strided_input ? "s" : "v");
        const std::string kernel_name = name + "_col_" + mode + "_" + dtype.str();
        encoder.set_pipeline_state(kernel_name);

        auto &view = input->get_view();
        uint32_t nrows = view[0];
        uint32_t ncols = view[1];
        // This threadgroup_size is applied to each row
        uint32_t max_threadgroup_size = encoder.get_kernel()->get_state()->maxTotalThreadsPerThreadgroup();
        uint32_t simd_size = encoder.get_kernel()->get_state()->threadExecutionWidth();
        uint32_t row_threadgroup_size = std::min(nrows, max_threadgroup_size / simd_size);
        uint32_t col_threadgroup_size = max_threadgroup_size / row_threadgroup_size;
        // Ensure threadgroup size is multiple of SIMD size
        col_threadgroup_size = (col_threadgroup_size / simd_size) * simd_size;
        // Set threadgroup memory size
        uint32_t threadgroup_nbytes = col_threadgroup_size * row_threadgroup_size * dtype.get_size();
        encoder.get_internal_encoder()->setThreadgroupMemoryLength(threadgroup_nbytes, 0);
        // Compute grid and threadgroup size
        MTL::Size grid_size = MTL::Size::Make(ncols, nrows, 1);
        MTL::Size threadgroup_size = MTL::Size::Make(col_threadgroup_size, row_threadgroup_size, 1);

        // Dispatch kernel
        encoder.dispatch_threads(grid_size, threadgroup_size);
        pool->release();
    }
}