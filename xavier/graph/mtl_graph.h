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
        std::unordered_set<IdType> visited;
        std::vector<std::shared_ptr<Array>> fw_order;
        std::vector<std::shared_ptr<Array>> bw_order;

        void toposort(std::shared_ptr<Array> arr, std::vector<std::shared_ptr<Array>> &order);

        void call(std::shared_ptr<Array> arr);

        void call_initializer(std::shared_ptr<Array> arr);

        void call_unary(const std::string &name, std::shared_ptr<Array> arr);

        void call_binary(const std::string &name, std::shared_ptr<Array> arr);

        void call_ibinary(const std::string &name, std::shared_ptr<Array> arr);

        void call_transform(std::shared_ptr<Array> arr);

    public:
        MTLGraph(std::shared_ptr<Array> root, std::shared_ptr<MTLContext> ctx) : Graph(root), ctx(ctx) {}

        void compile() override;

        void forward() override;

        void backward() override;

        const std::string str() const override;
    };
}