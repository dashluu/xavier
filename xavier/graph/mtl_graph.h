#pragma once

#include "../backend/metal/mtl_initializers.h"
#include "../backend/metal/mtl_copy.h"
#include "../backend/metal/mtl_unary.h"
#include "../backend/metal/mtl_binary.h"
#include "../backend/metal/mtl_matmul.h"
#include "../backend/metal/mtl_reduce.h"
#include "graph.h"

namespace xv::graph
{
    using namespace xv::backend::metal;

    class MTLGraph : public Graph
    {
    private:
        std::shared_ptr<MTLContext> ctx;
        std::unordered_set<Id> visited;
        std::vector<ArrayPtr> fw_order;
        std::vector<ArrayPtr> bw_order;

        void toposort(ArrayPtr arr, std::vector<ArrayPtr> &order);

        void call(ArrayPtr arr);

        void call_initializer(ArrayPtr arr);

        void call_unary(ArrayPtr arr);

        void call_binary(ArrayPtr arr);

        void call_transform(ArrayPtr arr);

        void call_reduce(ArrayPtr arr);

        void call_move(ArrayPtr arr);

    public:
        MTLGraph(ArrayPtr root, std::shared_ptr<MTLContext> ctx) : Graph(root), ctx(ctx) {}

        void compile() override;

        void forward() override;

        void backward() override;

        const std::string str() const override;
    };
}