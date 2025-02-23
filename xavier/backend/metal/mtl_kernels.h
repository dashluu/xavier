#pragma once

#include "mtl_context.h"

namespace xv::backend::metal
{
    void constant(std::shared_ptr<Array> arr, float c, MTLContext &ctx);

    void arange(std::shared_ptr<Array> arr, int start, int step, MTLContext &ctx);

    void copy(std::shared_ptr<Array> input, std::shared_ptr<Array> output, MTLContext &ctx);

    void sparse_copy(std::shared_ptr<Array> input, std::shared_ptr<Array> output, MTLContext &ctx);

    void ss_op(const std::string &name, std::vector<std::shared_ptr<Array>> input, std::shared_ptr<Array> output, MTLContext &ctx);

    void sparse_ss_op(const std::string &name, std::vector<std::shared_ptr<Array>> input, std::shared_ptr<Array> output, MTLContext &ctx);
}