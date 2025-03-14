#include "mtl_graph.h"

namespace xv::graph
{
    void MTLGraph::call_initializer(std::shared_ptr<Array> arr)
    {
        arr->alloc();
        auto op = arr->get_op();
        switch (op->get_name())
        {
        case OpName::FULL:
        {
            full(arr, std::static_pointer_cast<FullOp>(op)->get_const(), *ctx);
            break;
        }
        case OpName::ARANGE:
        {
            auto arange_op = std::static_pointer_cast<ArangeOp>(op);
            arange(arr, arange_op->get_start(), arange_op->get_step(), *ctx);
            break;
        }
        default:
            break;
        }
    }

    void MTLGraph::call_unary(const std::string &name, std::shared_ptr<Array> arr)
    {
        auto unary_op = std::static_pointer_cast<UnaryOp>(arr->get_op());
        auto operand = unary_op->get_operand();
        arr->alloc();
        std::vector<std::shared_ptr<Array>> input = {operand, arr};
        ss_op(name, input, *ctx);
    }

    void MTLGraph::call_binary(const std::string &name, std::shared_ptr<Array> arr)
    {
        auto binary_op = std::static_pointer_cast<BinaryOp>(arr->get_op());
        auto lhs = binary_op->get_lhs();
        auto rhs = binary_op->get_rhs();
        arr->alloc();
        std::vector<std::shared_ptr<Array>> input = {lhs, rhs, arr};
        ss_op(name, input, *ctx);
    }

    void MTLGraph::call_self_binary(const std::string &name, std::shared_ptr<Array> arr)
    {
        auto binary_op = std::static_pointer_cast<SelfBinaryOp>(arr->get_op());
        auto lhs = binary_op->get_lhs();
        auto rhs = binary_op->get_rhs();
        arr->alloc(*lhs->get_buff());
        std::vector<std::shared_ptr<Array>> input = {rhs, lhs};
        ss_op(name, input, *ctx);
    }

    void MTLGraph::call_transform(std::shared_ptr<Array> arr)
    {
        auto op = arr->get_op();
        auto transform_op = std::static_pointer_cast<TransformOp>(op);
        auto operand = transform_op->get_operand();
        switch (op->get_name())
        {
        case OpName::RESHAPE:
        {
            arr->alloc();
            sparse_copy(operand, arr, *ctx);
            break;
        }
        default:
            break;
        }
    }

    void MTLGraph::toposort(std::shared_ptr<Array> arr, std::vector<std::shared_ptr<Array>> &order)
    {
        if (visited.contains(arr->get_id()))
        {
            return;
        }
        visited.insert(arr->get_id());
        auto op = arr->get_op();
        switch (op->get_type())
        {
        case OpType::INITIALIZER:
        {
            order.push_back(arr);
            break;
        }
        case OpType::UNARY:
        {
            auto unary_op = std::static_pointer_cast<UnaryOp>(arr->get_op());
            auto operand = unary_op->get_operand();
            toposort(operand, order);
            order.push_back(arr);
            break;
        }
        case OpType::BINARY:
        {
            auto binary_op = std::static_pointer_cast<BinaryOp>(arr->get_op());
            auto lhs = binary_op->get_lhs();
            auto rhs = binary_op->get_rhs();
            toposort(lhs, order);
            toposort(rhs, order);
            order.push_back(arr);
            break;
        }
        case OpType::SELF_BINARY:
        {
            auto binary_op = std::static_pointer_cast<SelfBinaryOp>(arr->get_op());
            auto lhs = binary_op->get_lhs();
            auto rhs = binary_op->get_rhs();
            toposort(lhs, order);
            toposort(rhs, order);
            order.push_back(arr);
            break;
        }
        case OpType::TRANSFORM:
        {
            auto transform_op = std::static_pointer_cast<TransformOp>(arr->get_op());
            auto operand = transform_op->get_operand();
            toposort(operand, order);
            order.push_back(arr);
            break;
        }
        default:
            break;
        }
    }

    void MTLGraph::call(std::shared_ptr<Array> arr)
    {
        auto op = arr->get_op();
        switch (op->get_type())
        {
        case OpType::INITIALIZER:
        {
            call_initializer(arr);
            break;
        }
        case OpType::UNARY:
        {
            call_unary(opnames.at(op->get_name()), arr);
            break;
        }
        case OpType::BINARY:
        {
            call_binary(opnames.at(op->get_name()), arr);
            break;
        }
        case OpType::SELF_BINARY:
        {
            call_self_binary(opnames.at(op->get_name()), arr);
            break;
        }
        case OpType::TRANSFORM:
        {
            call_transform(arr);
            break;
        }
        default:
            break;
        }
    }

    void MTLGraph::compile()
    {
        if (fw_order.empty())
        {
            toposort(root, fw_order);
            // Seed root for now
            root->grad = Array::ones_like(root);
            // Initializes the gradient array first without allocating buffers
            for (auto &arr : std::views::reverse(fw_order))
            {
                arr->get_op()->backward(arr);
            }
            // Order the gradient arrays
            for (auto &arr : std::views::reverse(fw_order))
            {
                if (arr->grad != nullptr)
                {
                    toposort(arr->grad, bw_order);
                }
            }
        }
    }

    void MTLGraph::forward()
    {
        if (fw_order.empty())
        {
            throw MTLGraphNotCompiledException();
        }
        for (auto &arr : fw_order)
        {
            call(arr);
        }
    }

    void MTLGraph::backward()
    {
        if (bw_order.empty())
        {
            throw MTLGraphNotCompiledException();
        }
        for (auto &arr : bw_order)
        {
            call(arr);
        }
    }

    const std::string MTLGraph::str() const
    {
        if (fw_order.empty())
        {
            throw MTLGraphNotCompiledException();
        }
        std::string s = "Forward:\n";
        for (auto &arr : fw_order)
        {
            s += std::to_string(arr->get_id()) + ": " + arr->get_op()->str() + "\n";
        }
        s += "Backward:\n";
        for (auto &arr : bw_order)
        {
            s += std::to_string(arr->get_id()) + ": " + arr->get_op()->str() + "\n";
        }
        return s;
    }
}