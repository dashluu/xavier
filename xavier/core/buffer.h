#pragma once

#include "../common.h"

namespace xv::core
{
    struct Buffer : std::enable_shared_from_this<Buffer>
    {
    private:
        uint8_t *ptr;
        usize nbytes;
        bool root;

    public:
        Buffer(uint8_t *ptr, usize nbytes, bool root) : ptr(ptr), nbytes(nbytes), root(root)
        {
        }

        Buffer(const Buffer &buff) : ptr(buff.ptr), nbytes(buff.nbytes), root(false) {}

        Buffer &operator=(const Buffer &buff)
        {
            ptr = buff.ptr;
            nbytes = buff.nbytes;
            root = false;
            return *this;
        }

        uint8_t *get_ptr() const { return ptr; }

        usize get_nbytes() const { return nbytes; }

        bool is_root() { return root; }
    };
}