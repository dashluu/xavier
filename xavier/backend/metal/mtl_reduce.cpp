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

        // Configure kernel
        auto dtype = input->get_dtype();
        const std::string mode = "v" + std::string(strided_input ? "s" : "v");
        const std::string kernel_name = name + "_all_" + mode + "_" + dtype.str();
        encoder.set_pipeline_state(kernel_name);

        // Calculate optimal thread configuration
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
        // Initialize Metal autorelease pool and encoder
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

        // Configure kernel
        const Dtype &dtype = input->get_dtype();
        const std::string mode = "v" + std::string(strided_input ? "s" : "v");
        const std::string kernel_name = name + "_col_" + mode + "_" + dtype.str();
        encoder.set_pipeline_state(kernel_name);

        // Calculate optimal thread configuration
        const usize max_threadgroup_size = encoder.get_kernel()->get_state()->maxTotalThreadsPerThreadgroup();
        const usize simd_size = encoder.get_kernel()->get_state()->threadExecutionWidth();
        const ShapeView &view = input->get_view();
        usize nrows = view[0];
        usize ncols = align_to(view[1], simd_size);
        const usize col_threadgroup_size = std::min(ncols, max_threadgroup_size);
        const usize row_threadgroup_size = std::min(nrows, max_threadgroup_size / col_threadgroup_size);
        MTL::Size grid_size = MTL::Size::Make(ncols, nrows, 1);
        MTL::Size threadgroup_size = MTL::Size::Make(col_threadgroup_size, row_threadgroup_size, 1);
        const usize threadgroup_nbytes = col_threadgroup_size * row_threadgroup_size * dtype.get_size();
        encoder.get_internal_encoder()->setThreadgroupMemoryLength(threadgroup_nbytes, 0);

        // Dispatch kernel
        encoder.dispatch_threads(grid_size, threadgroup_size);
        pool->release();
    }
}