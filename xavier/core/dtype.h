#pragma once

#include "../common.h"

namespace xv::core
{
    class Dtype
    {
    private:
        std::string name;
        usize size;

    public:
        Dtype(const std::string &name, usize size) : name(name), size(size) {}

        Dtype(const Dtype &dtype) : Dtype(dtype.name, dtype.size) {}

        const std::string &get_name() const { return name; }

        usize get_size() const { return size; }

        bool operator==(const Dtype &dtype) const { return name == dtype.name && size == dtype.size; }

        bool operator!=(const Dtype &dtype) const { return !(*this == dtype); }

        Dtype &operator=(const Dtype &dtype)
        {
            name = dtype.name;
            size = dtype.size;
            return *this;
        }

        std::string str() const
        {
            return name;
        }
    };

    inline const Dtype f16("f16", 2);
    inline const Dtype f32("f32", 4);
    inline const Dtype i8("i8", 1);
    inline const Dtype i16("i16", 2);
    inline const Dtype i32("i32", 4);
    inline const Dtype b8("b8", 1);
}

namespace std
{
    template <>
    struct hash<xv::core::Dtype>
    {
        std::size_t operator()(const xv::core::Dtype &dtype) const
        {
            return std::hash<std::string>()(dtype.get_name());
        }
    };
}

namespace xv::core
{
    inline const std::unordered_set<Dtype> all_dtypes = {b8, i32, f32};
    inline const std::unordered_set<Dtype> num_dtypes = {i32, f32};
    inline const std::unordered_set<Dtype> bool_dtypes = {b8};
    inline const std::unordered_set<Dtype> int_dtypes = {i32};
    inline const std::unordered_set<Dtype> float_dtypes = {f32};
    inline const std::unordered_set<Dtype> binary_dtypes = {i32, f32};
    inline const std::unordered_set<Dtype> unary_dtypes = {i32, f32};
    inline const std::unordered_map<Dtype, Dtype> unary_float_dtypes = {
        {i32, f32},
        {f32, f32}};
}