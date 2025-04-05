#include "mtl_binary.h"

namespace xv::backend::metal
{
    void binary_ss(const std::string &name, ArrayPtr lhs, ArrayPtr rhs, ArrayPtr output, std::shared_ptr<MTLContext> ctx)
    {
        NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
        CommandEncoder encoder(ctx);
        bool strided_input = !lhs->is_contiguous() || !rhs->is_contiguous();
        bool strided_output = !output->is_contiguous();
        if (strided_input || strided_output)
        {
            encoder.encode_ndim(lhs);
        }
        encoder.encode_offset({lhs, rhs, output});
        if (strided_input || strided_output)
        {
            encoder.encode_view(lhs);
        }
        if (strided_input)
        {
            encoder.encode_stride(lhs);
            encoder.encode_stride(rhs);
        }
        if (strided_output)
        {
            encoder.encode_stride(output);
        }
        encoder.encode_array(lhs);
        encoder.encode_array(rhs);
        encoder.encode_array(output);
        const std::string mode = std::string(strided_output ? "s" : "v") + std::string(strided_input ? "s" : "v");
        const std::string kernel_name = name + "_" + mode + "_" + lhs->get_dtype().str();
        encoder.set_pipeline_state(kernel_name);
        encoder.dispatch_threads(lhs->get_numel());
        pool->release();
    }
}