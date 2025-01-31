#include "graph.h"

namespace xv::core
{
    const std::string Graph::recur_str(std::shared_ptr<Array> arr, std::unordered_set<IdType> &visited) const
    {
        if (visited.contains(arr->get_id()))
        {
            return "";
        }
        visited.insert(arr->get_id());
        auto op = arr->get_op();
        auto s = std::to_string(arr->get_id()) + ": ";
        switch (op->get_type())
        {
        case OpType::INITIALIZER:
            return s + op->str() + "\n";
        case OpType::UNARY:
        {
            auto unary_op = std::static_pointer_cast<UnaryOp>(op);
            auto operand = unary_op->get_operand();
            return s + unary_op->str() + "\n" + recur_str(operand, visited);
        }
        case OpType::BINARY:
        {
            auto binary_op = std::static_pointer_cast<BinaryOp>(op);
            auto lhs = binary_op->get_lhs();
            auto rhs = binary_op->get_rhs();
            return s + binary_op->str() + "\n" + recur_str(lhs, visited) + recur_str(rhs, visited);
        }
        case OpType::TRANSFORM:
        {
            auto transform_op = std::static_pointer_cast<TransformOp>(op);
            auto operand = transform_op->get_operand();
            return s + transform_op->str() + "\n" + recur_str(operand, visited);
        }
        default:
            return "";
        }
    }

    const std::string Graph::str() const
    {
        std::unordered_set<IdType> visited;
        return recur_str(root, visited);
    }
}