#pragma once

#include "array.h"

namespace xv::core
{
    struct ArrayIter
    {
    private:
        std::shared_ptr<const Array> arr;
        uint8_t *ptr;
        uint64_t counter;

    public:
        ArrayIter(std::shared_ptr<const Array> arr) : arr(arr)
        {
        }

        ArrayIter(const ArrayIter &) = delete;

        ArrayIter &operator=(const ArrayIter &) = delete;

        bool has_next() const { return counter < arr->get_shape().get_numel(); }

        uint64_t count() const { return counter; }

        void start()
        {
            counter = 0;
        }

        uint8_t *next()
        {
            ptr = arr->strided_idx(counter);
            counter++;
            return ptr;
        }
    };
}