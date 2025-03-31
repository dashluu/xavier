#pragma once

#include <iostream>
#include <sstream>
#include <algorithm>
#include <vector>
#include <string>
#include <ranges>
#include <memory>
#include <iomanip>
#include <numeric>
#include <cstdlib>
#include <cmath>
#include <unordered_set>
#include <unordered_map>
#include <functional>
#include <type_traits>

namespace xv::core
{
    class Array;
    using ArrayPtr = std::shared_ptr<Array>;

    class IStr
    {
    public:
        virtual const std::string str() const = 0;
    };

    template <class T>
    inline const std::string vstr(const std::vector<T> &v, const std::function<std::string(T)> &f)
    {
        std::string s = "";
        for (int i = 0; i < v.size(); i++)
        {
            s += f(v[i]);
            if (i < v.size() - 1)
            {
                s += ", ";
            }
        }
        return s;
    }

    template <class T>
    inline const std::string vnumstr(const std::vector<T> &v)
    {
        return vstr<T>(v, [](T a)
                       { return std::to_string(a); });
    }
}