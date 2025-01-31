#pragma once

#include "../common.h"

namespace xv::core
{
    struct Range
    {
        uint64_t start;
        uint64_t stop;
        int64_t step;

        Range(uint64_t start, uint64_t stop, int64_t step = 1) : start(start), stop(stop), step(step) {}

        Range(const Range &range) : Range(range.start, range.stop, range.step) {}

        Range &operator=(const Range &range)
        {
            start = range.start;
            stop = range.stop;
            step = range.step;
            return *this;
        }

        bool operator==(const Range &range) const
        {
            return start == range.start && stop == range.stop && step == range.step;
        }

        std::string str() const
        {
            return "(" + std::to_string(start) + ", " + std::to_string(stop) + ", " + std::to_string(step) + ")";
        }
    };
}