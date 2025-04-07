#pragma once

#include "../common.h"

namespace xv::core
{
    struct Range
    {
        isize start;
        // Stop is exclusive
        isize stop;
        isize step;

        Range(isize start, isize stop, isize step = 1) : start(start), stop(stop), step(step) {}

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

    using Ranges = std::vector<Range>;
}