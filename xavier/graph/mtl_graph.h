#pragma once

#include "../backend/metal/mtl_kernels.h"
#include "graph.h"

namespace xv::graph
{
    using namespace xv::backend::metal;

    class MTLGraph : public Graph
    {
    private:
        std::shared_ptr<MTLContext> ctx;
        std::vector<std::shared_ptr<Array>> leaves;

        void initializer(std::shared_ptr<Array> arr);

        void unary(const std::string &name, std::shared_ptr<Array> arr, std::unordered_set<IdType> &visited);

        void binary(const std::string &name, std::shared_ptr<Array> arr, std::unordered_set<IdType> &visited);

        void transform(std::shared_ptr<Array> arr, std::unordered_set<IdType> &visited);

        void recur_forward(std::shared_ptr<Array> arr, std::unordered_set<IdType> &visited);

        void recur_backward(std::shared_ptr<Array> arr, std::unordered_set<IdType> &visited);

    public:
        MTLGraph(std::shared_ptr<Array> root, std::shared_ptr<MTLContext> ctx) : Graph(root), ctx(ctx) {}

        void forward() override;

        void backward() override;
    };
}