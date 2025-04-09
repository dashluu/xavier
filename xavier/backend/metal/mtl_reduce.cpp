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
        // Initialize Metal autorelease pool and encoder
        NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
        CommandEncoder copy_encoder(ctx);
        bool strided_input = !input->is_contiguous();

        // First pass: Copy and reorganize input data
        if (strided_input)
        {
            copy_encoder.encode_ndim(input);
        }
        const Dtype &dtype = input->get_dtype();
        const ShapeView &view = input->get_view();
        usize nrows = view[0];
        usize ncols = view[1];
        const std::string mode = "v" + std::string(strided_input ? "s" : "v");
        const std::string copy_kernel_name = "reduce_col_copy_" + mode + "_" + dtype.str();
        copy_encoder.set_pipeline_state(copy_kernel_name);

        // Calculate optimal thread configuration
        const usize max_threadgroup_size = copy_encoder.get_kernel()->get_state()->maxTotalThreadsPerThreadgroup();
        const usize simd_size = copy_encoder.get_kernel()->get_state()->threadExecutionWidth();
        ncols = align_to(ncols, simd_size);

        // Allocate temporary buffer for intermediate results
        ArrayPtr tmp_buff = std::make_shared<Array>(Shape({nrows, ncols}), dtype, input->get_device());
        tmp_buff->alloc();

        // Set up encoder parameters for copy operation
        copy_encoder.encode_offset({input});
        copy_encoder.encode_view(input);
        copy_encoder.encode_view(tmp_buff);
        if (strided_input)
        {
            copy_encoder.encode_stride(input);
        }
        copy_encoder.encode_array(input);
        copy_encoder.encode_array(tmp_buff);

        // Configure and dispatch copy kernel
        const usize col_threadgroup_size = std::min(ncols, max_threadgroup_size);
        const usize row_threadgroup_size = std::min(nrows, max_threadgroup_size / col_threadgroup_size);
        MTL::Size grid_size = MTL::Size::Make(ncols, nrows, 1);
        MTL::Size threadgroup_size = MTL::Size::Make(col_threadgroup_size, row_threadgroup_size, 1);
        copy_encoder.dispatch_threads(grid_size, threadgroup_size);

        // Second pass: Perform column reduction
        CommandEncoder reduce_encoder(ctx);
        reduce_encoder.encode_offset({output});
        reduce_encoder.encode_view(tmp_buff);
        reduce_encoder.encode_array(tmp_buff);
        reduce_encoder.encode_array(output);
        // Set threadgroup memory size
        const usize threadgroup_nbytes = col_threadgroup_size * row_threadgroup_size * dtype.get_size();
        reduce_encoder.get_internal_encoder()->setThreadgroupMemoryLength(threadgroup_nbytes, 0);

        // Dispatch reduction kernel
        const std::string reduce_kernel_name = name + "_col_" + dtype.str();
        reduce_encoder.set_pipeline_state(reduce_kernel_name);
        reduce_encoder.dispatch_threads(grid_size, threadgroup_size);
        pool->release();
    }
}