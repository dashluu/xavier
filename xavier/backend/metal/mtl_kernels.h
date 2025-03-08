#pragma once

#include "mtl_context.h"

namespace xv::backend::metal
{
    template <class T1, class T2>
    std::vector<T2> vec64to32(const std::vector<T1> &v)
    {
        std::vector<T2> v32(v.size());
        for (size_t i = 0; i < v.size(); i++)
        {
            v32[i] = static_cast<T2>(v[i]);
        }
        return v32;
    }

    void full(std::shared_ptr<Array> arr, float c, MTLContext &ctx);

    void arange(std::shared_ptr<Array> arr, int start, int step, MTLContext &ctx);

    void copy(std::shared_ptr<Array> input, std::shared_ptr<Array> output, MTLContext &ctx);

    void sparse_copy(std::shared_ptr<Array> input, std::shared_ptr<Array> output, MTLContext &ctx);

    void unary_ss(const std::string &name, std::shared_ptr<Array> input, std::shared_ptr<Array> output, MTLContext &ctx);

    void binary_ss(const std::string &name, std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs, std::shared_ptr<Array> output, MTLContext &ctx);

    void sparse_unary_ss(const std::string &name, std::shared_ptr<Array> input, std::shared_ptr<Array> output, MTLContext &ctx);

    void sparse_binary_ss(const std::string &name, std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs, std::shared_ptr<Array> output, MTLContext &ctx);
}