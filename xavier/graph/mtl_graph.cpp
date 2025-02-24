#include "mtl_graph.h"

namespace xv::graph
{
    void MTLGraph::initializer(std::shared_ptr<Array> arr)
    {
        arr->alloc();
        auto op = arr->get_op();
        switch (op->get_name())
        {
        case OpName::CONSTANT:
        {
            constant(arr, std::static_pointer_cast<ConstOp>(op)->get_const(), *ctx);
            break;
        }
        case OpName::ARANGE:
        {
            auto arange_op = std::static_pointer_cast<ArangeOp>(op);
            arange(arr, arange_op->get_start(), arange_op->get_step(), *ctx);
            break;
        }
        }
    }

    void MTLGraph::unary(const std::string &name, std::shared_ptr<Array> arr, std::unordered_set<IdType> &visited)
    {
        auto unary_op = std::static_pointer_cast<UnaryOp>(arr->get_op());
        auto operand = unary_op->get_operand();
        recur_forward(operand, visited);
        arr->alloc();
        std::vector<std::shared_ptr<Array>> input = {operand};
        ss_op(name, input, arr, *ctx);
    }

    void MTLGraph::binary(const std::string &name, std::shared_ptr<Array> arr, std::unordered_set<IdType> &visited)
    {
        auto binary_op = std::static_pointer_cast<BinaryOp>(arr->get_op());
        auto lhs = binary_op->get_lhs();
        auto rhs = binary_op->get_rhs();
        recur_forward(lhs, visited);
        recur_forward(rhs, visited);
        arr->alloc();
        std::vector<std::shared_ptr<Array>> input = {lhs, rhs};
        ss_op(name, input, arr, *ctx);
    }

    void MTLGraph::transform(std::shared_ptr<Array> arr, std::unordered_set<IdType> &visited)
    {
        auto op = arr->get_op();
        auto transform_op = std::static_pointer_cast<TransformOp>(op);
        auto operand = transform_op->get_operand();
        recur_forward(operand, visited);
        switch (op->get_name())
        {
        case OpName::RESHAPE:
        {
            arr->alloc();
            sparse_copy(operand, arr, *ctx);
            break;
        }
        }
    }

    void MTLGraph::recur_forward(std::shared_ptr<Array> arr, std::unordered_set<IdType> &visited)
    {
        if (visited.contains(arr->get_id()))
        {
            return;
        }
        auto op = arr->get_op();
        switch (op->get_type())
        {
        case OpType::INITIALIZER:
            initializer(arr);
            break;
        case OpType::UNARY:
            unary(opnames.at(op->get_name()), arr, visited);
            break;
        case OpType::BINARY:
            binary(opnames.at(op->get_name()), arr, visited);
            break;
        case OpType::TRANSFORM:
            transform(arr, visited);
            break;
        }
    }

    void MTLGraph::forward()
    {
        std::unordered_set<IdType> visited;
        recur_forward(root, visited);
    }

    void MTLGraph::backward()
    {
    }
}