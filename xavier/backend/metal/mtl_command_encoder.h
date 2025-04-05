#pragma once

#include "mtl_context.h"

namespace xv::backend::metal
{
    struct CommandEncoder
    {
    private:
        std::shared_ptr<MTLContext> ctx;
        MTL::CommandBuffer *cmd_buff;
        MTL::ComputeCommandEncoder *encoder;
        uint32_t buff_idx = 0;
        uint32_t ndim;
        uint32_t *offset = nullptr;
        std::vector<uint32_t *> views;
        std::vector<int32_t *> strides;
        std::shared_ptr<MTLKernel> kernel;

        template <class T1, class T2>
        T2 *vcast(const std::vector<T1> &v1)
        {
            T2 *v2 = new T2[v1.size()];
            for (size_t i = 0; i < v1.size(); i++)
            {
                v2[i] = static_cast<T2>(v1[i]);
            }
            return v2;
        }

    public:
        CommandEncoder(std::shared_ptr<MTLContext> ctx) : ctx(ctx)
        {
            cmd_buff = ctx->get_cmd_queue()->commandBuffer();
            encoder = cmd_buff->computeCommandEncoder();
        }

        ~CommandEncoder()
        {
            delete offset;
            for (auto view : views)
            {
                delete view;
            }
            for (auto stride : strides)
            {
                delete stride;
            }
        }

        std::shared_ptr<MTLKernel> get_kernel() { return kernel; }

        MTL::ComputeCommandEncoder *get_internal_encoder() { return encoder; }

        void encode_buffer(const void *buff, usize size)
        {
            MTL::Buffer *mtl_buff = ctx->get_device()->newBuffer(buff, size, MTL::ResourceStorageModeShared, nullptr);
            encoder->setBuffer(mtl_buff, 0, buff_idx++);
        }

        void encode_ndim(ArrayPtr arr)
        {
            ndim = static_cast<uint32_t>(arr->get_ndim());
            encode_buffer(&ndim, sizeof(uint32_t));
        }

        void encode_offset(const std::vector<ArrayPtr> arrs)
        {
            offset = new uint32_t[arrs.size()];
            for (size_t i = 0; i < arrs.size(); i++)
            {
                offset[i] = static_cast<uint32_t>(arrs[i]->get_offset());
            }
            encode_buffer(offset, sizeof(offset));
        }

        void encode_view(ArrayPtr arr)
        {
            uint32_t *view = vcast<usize, uint32_t>(arr->get_view());
            encode_buffer(view, sizeof(view));
            views.push_back(view);
        }

        void encode_stride(ArrayPtr arr)
        {
            int32_t *stride = vcast<isize, int32_t>(arr->get_stride());
            encode_buffer(stride, sizeof(stride));
            strides.push_back(stride);
        }

        void encode_array(ArrayPtr arr)
        {
            encode_buffer(arr->get_buff_ptr(), arr->get_buff_nbytes());
        }

        void set_pipeline_state(const std::string &kernel_name)
        {
            kernel = ctx->get_kernel(kernel_name);
            encoder->setComputePipelineState(kernel->get_state().get());
        }

        void dispatch_threads(uint64_t nthreads)
        {
            auto grid_size = MTL::Size::Make(nthreads, 1, 1);
            auto threadgroup_size = MTL::Size::Make(kernel->get_state()->maxTotalThreadsPerThreadgroup(), 1, 1);
            dispatch_threads(grid_size, threadgroup_size);
        }

        void dispatch_threads(MTL::Size grid_size, MTL::Size threadgroup_size)
        {
            encoder->dispatchThreads(grid_size, threadgroup_size);
            encoder->endEncoding();
            cmd_buff->commit();
            cmd_buff->waitUntilCompleted();
        }

        void dispatch_threadgroups(MTL::Size threadgroup_count, MTL::Size threadgroup_size)
        {
            encoder->dispatchThreadgroups(threadgroup_count, threadgroup_size);
            encoder->endEncoding();
            cmd_buff->commit();
            cmd_buff->waitUntilCompleted();
        }
    };
}