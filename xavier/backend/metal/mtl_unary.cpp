#include "mtl_unary.h"

namespace xv::backend::metal
{
    void unary_ss(const std::string &name, ArrayPtr input, ArrayPtr output, std::shared_ptr<MTLContext> ctx)
    {
        NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
        CommandEncoder encoder(ctx);
        bool strided_input = !input->is_contiguous();
        bool strided_output = !output->is_contiguous();
        if (strided_input || strided_output)
        {
            encoder.encode_ndim(input);
        }
        encoder.encode_offset({input, output});
        if (strided_input || strided_output)
        {
            encoder.encode_view(input);
        }
        if (strided_input)
        {
            encoder.encode_stride(input);
        }
        if (strided_output)
        {
            encoder.encode_stride(output);
        }
        encoder.encode_array(input);
        encoder.encode_array(output);
        const std::string mode = std::string(strided_output ? "s" : "v") + std::string(strided_input ? "s" : "v");
        const std::string kernel_name = name + "_" + mode + "_" + input->get_dtype().str();
        encoder.set_pipeline_state(kernel_name);
        encoder.dispatch_threads(input->get_numel());
        pool->release();
    }
}