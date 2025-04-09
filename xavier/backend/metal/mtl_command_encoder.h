#pragma once

#include "mtl_context.h"

namespace xv::backend::metal
{
    struct CommandEncoder
    {
    private:
        using mtl_usize = uint32_t;
        using mtl_isize = int32_t;
        std::shared_ptr<MTLContext> ctx;
        MTL::CommandBuffer *cmd_buff;
        MTL::ComputeCommandEncoder *encoder;
        usize buff_idx = 0;
        std::vector<uint8_t *> buffs;
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
            for (auto buff : buffs)
            {
                delete[] buff;
            }
        }

        std::shared_ptr<MTLKernel> get_kernel() { return kernel; }

        MTL::ComputeCommandEncoder *get_internal_encoder() { return encoder; }

        template <class T>
        void encode_scalar(T scalar)
        {
            T *scalar_buff = new T[1];
            scalar_buff[0] = scalar;
            encode_buffer(scalar_buff, sizeof(T), true);
        }

        void encode_buffer(void *buff, usize size, bool mem_tracked)
        {
            MTL::Buffer *mtl_buff = ctx->get_device()->newBuffer(buff, size, MTL::ResourceStorageModeShared, nullptr);
            encoder->setBuffer(mtl_buff, 0, buff_idx++);
            if (mem_tracked)
            {
                buffs.push_back(static_cast<uint8_t *>(buff));
            }
        }

        void encode_ndim(ArrayPtr arr)
        {
            mtl_usize ndim = static_cast<uint32_t>(arr->get_ndim());
            encode_scalar(ndim);
        }

        void encode_offset(const std::vector<ArrayPtr> arrs)
        {
            mtl_usize *offset = new mtl_usize[arrs.size()];
            for (size_t i = 0; i < arrs.size(); i++)
            {
                offset[i] = static_cast<mtl_usize>(arrs[i]->get_offset());
            }
            encode_buffer(offset, sizeof(mtl_usize) * arrs.size(), true);
        }

        void encode_view(ArrayPtr arr)
        {
            mtl_usize *view = vcast<usize, mtl_usize>(arr->get_view());
            encode_buffer(view, sizeof(mtl_usize) * arr->get_ndim(), true);
        }

        void encode_stride(ArrayPtr arr)
        {
            mtl_isize *stride = vcast<isize, mtl_isize>(arr->get_stride());
            encode_buffer(stride, sizeof(mtl_isize) * arr->get_ndim(), true);
        }

        void encode_array(ArrayPtr arr)
        {
            encode_buffer(arr->get_buff_ptr(), arr->get_buff_nbytes(), false);
        }

        void set_pipeline_state(const std::string &kernel_name)
        {
            kernel = ctx->get_kernel(kernel_name);
            encoder->setComputePipelineState(kernel->get_state().get());
        }

        void dispatch_threads(usize nthreads)
        {
            MTL::Size grid_size = MTL::Size::Make(nthreads, 1, 1);
            usize max_threadgroup_size = kernel->get_state()->maxTotalThreadsPerThreadgroup();
            MTL::Size threadgroup_size = MTL::Size::Make(std::min(nthreads, max_threadgroup_size), 1, 1);
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