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
        std::vector<uint64_t> idx;

        std::vector<uint64_t> gen_idx()
        {
            std::vector<uint64_t> next = idx;
            auto &shape = arr->get_shape();
            int i = shape.get_ndim() - 1;
            bool flag = false;
            while (i >= 0 && !flag)
            {
                if (next[i] + 1 < shape[i])
                {
                    next[i]++;
                    flag = true;
                }
                else
                {
                    next[i] = 0;
                    i--;
                }
            }
            return next;
        }

    public:
        ArrayIter(std::shared_ptr<const Array> arr) : arr(arr)
        {
        }

        ArrayIter(const ArrayIter &) = delete;

        ArrayIter &operator=(const ArrayIter &) = delete;

        bool has_next() const { return counter <= arr->get_shape().get_numel(); }

        uint8_t *curr() const { return ptr; }

        uint64_t count() const { return counter; }

        void start()
        {
            idx = std::vector<uint64_t>(arr->get_shape().get_ndim(), 0);
            ptr = arr->get_ptr();
            counter = 1;
        }

        void next()
        {
            counter++;
            if (counter > arr->get_shape().get_numel())
            {
                return;
            }
            idx = gen_idx();
            ptr = arr->get_ptr();
            auto shape = arr->get_shape();
            auto stride = shape.get_stride();
            for (size_t i = 0; i < idx.size(); i++)
            {
                ptr += idx[i] * stride[i] * arr->get_dtype().get_size();
            }
        }
    };
}