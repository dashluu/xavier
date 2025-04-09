#include "mtl_initializers.h"

namespace xv::backend::metal
{
    void full(ArrayPtr arr, int c, usize size, std::shared_ptr<MTLContext> ctx)
    {
        NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
        CommandEncoder encoder(ctx);
        encoder.encode_buffer(&c, size, false);
        encoder.encode_array(arr);
        const std::string kernel_name = "full_" + arr->get_dtype().str();
        encoder.set_pipeline_state(kernel_name);
        encoder.dispatch_threads(arr->get_numel());
        pool->release();
    }

    void arange(ArrayPtr arr, int start, int step, std::shared_ptr<MTLContext> ctx)
    {
        NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
        CommandEncoder encoder(ctx);
        encoder.encode_buffer(&start, sizeof(start), false);
        encoder.encode_buffer(&step, sizeof(step), false);
        encoder.encode_array(arr);
        const std::string kernel_name = "arange_" + arr->get_dtype().str();
        encoder.set_pipeline_state(kernel_name);
        encoder.dispatch_threads(arr->get_numel());
        pool->release();
    }
}