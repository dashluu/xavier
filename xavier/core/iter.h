#pragma once

#include "array.h"

namespace xv::core
{
    struct ArrayIter
    {
    private:
        std::shared_ptr<const Array> arr;
        uint8_t *ptr;
        usize counter;

    public:
        ArrayIter(std::shared_ptr<const Array> arr) : arr(arr)
        {
        }

        ArrayIter(const ArrayIter &) = delete;

        ArrayIter &operator=(const ArrayIter &) = delete;

        bool has_next() const { return counter < arr->get_shape().get_numel(); }

        usize count() const { return counter; }

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