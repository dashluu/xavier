#include "mtl_matmul.h"

namespace xv::backend::metal
{
    void matmul(ArrayPtr lhs, ArrayPtr rhs, ArrayPtr output, std::shared_ptr<MTLContext> ctx)
    {
        NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
        CommandEncoder encoder(ctx);
        bool strided_input = !lhs->is_contiguous() || !rhs->is_contiguous();

        // Encode buffers
        if (strided_input)
        {
            encoder.encode_ndim(lhs);
        }
        encoder.encode_offset({lhs, rhs, output});
        encoder.encode_view(lhs);
        encoder.encode_view(rhs);
        if (strided_input)
        {
            encoder.encode_stride(lhs);
            encoder.encode_stride(rhs);
        }
        encoder.encode_array(lhs);
        encoder.encode_array(rhs);
        encoder.encode_array(output);
        const std::string mode = "v" + std::string(strided_input ? "s" : "v");
        const std::string kernel_name = "matmul_" + mode + "_" + lhs->get_dtype().str();
        encoder.set_pipeline_state(kernel_name);

        auto lhs_view = lhs->get_view();
        auto rhs_view = rhs->get_view();
        const uint32_t B = lhs_view[0]; // Batch size
        const uint32_t M = lhs_view[1]; // Number of rows
        const uint32_t K = lhs_view[2]; // Inner dimension
        const uint32_t N = rhs_view[2]; // Number of columns
        // Even if matrix is smaller than one threadgroup, we still need at least 1 group
        uint64_t x_group_count = std::max(1ull, static_cast<uint64_t>((N + X_THREADS_PER_GROUP - 1) / X_THREADS_PER_GROUP));
        uint64_t y_group_count = std::max(1ull, static_cast<uint64_t>((M + Y_THREADS_PER_GROUP - 1) / Y_THREADS_PER_GROUP));
        uint64_t z_group_count = std::max(1ull, static_cast<uint64_t>((B + Z_THREADS_PER_GROUP - 1) / Z_THREADS_PER_GROUP));
        // Compute # threadgroups and threadgroup size
        auto threadgroup_count = MTL::Size::Make(x_group_count, y_group_count, z_group_count);
        auto threadgroup_size = MTL::Size::Make(X_THREADS_PER_GROUP, Y_THREADS_PER_GROUP, Z_THREADS_PER_GROUP);

        // Dispatch kernel
        encoder.dispatch_threadgroups(threadgroup_count, threadgroup_size);
        pool->release();
    }
}
