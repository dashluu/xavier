#pragma once

#include "../common.h"
#include "buffer.h"

namespace xv::core
{

    struct Allocator
    {
    protected:
        usize allocated = 0;

    public:
        Allocator() = default;
        Allocator(const Allocator &) = delete;
        Allocator &operator=(const Allocator &) = delete;
        virtual std::shared_ptr<Buffer> alloc(usize nbytes) = 0;
        virtual void free(std::shared_ptr<Buffer> buff) = 0;
        usize get_allocated() const { return allocated; }
    };

    struct CommonAllocator : public Allocator
    {
        std::shared_ptr<Buffer> alloc(usize nbytes) override
        {
            allocated += nbytes;
            auto ptr = new uint8_t[nbytes];
            return std::make_shared<Buffer>(ptr, nbytes, true);
        }

        void free(std::shared_ptr<Buffer> buff) override
        {
            if (buff->is_root())
            {
                allocated -= buff->get_nbytes();
                delete[] buff->get_ptr();
            }
        }
    };

#if __APPLE__
    // Allocator for both CPU and MPS
    inline Allocator *allocator0 = new CommonAllocator();
#else
    // TODO: implement for other platforms such as CUDA
#endif
}