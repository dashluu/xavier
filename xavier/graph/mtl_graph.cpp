#include "mtl_graph.h"

namespace xv::graph
{
    void MTLGraph::call_initializer(ArrayPtr arr)
    {
        arr->alloc();
        auto op = arr->get_op();
        switch (op->get_name())
        {
        case OpName::FULL:
        {
            auto full_op = std::static_pointer_cast<FullOp>(op);
            full(arr, full_op->get_const(), arr->get_dtype().get_size(), *ctx);
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

    void MTLGraph::call_unary(ArrayPtr arr)
    {
        auto unary_op = std::static_pointer_cast<UnaryOp>(arr->get_op());
        auto operand = unary_op->get_operand();
        if (unary_op->is_in_place())
        {
            arr->alloc(*operand->get_buff());
        }
        else
        {
            arr->alloc();
        }
        if (operand->is_contiguous())
        {
            unary_ss(unary_op->get_name_str(), operand, arr, *ctx);
        }
        else
        {
            strided_unary_ss(unary_op->get_name_str(), operand, arr, *ctx);
        }
    }

    void MTLGraph::call_binary(ArrayPtr arr)
    {
        auto binary_op = std::static_pointer_cast<BinaryOp>(arr->get_op());
        auto lhs = binary_op->get_lhs();
        auto rhs = binary_op->get_rhs();
        if (binary_op->get_name() == OpName::MATMUL)
        {
            arr->alloc();
            if (lhs->is_contiguous() && rhs->is_contiguous())
            {
                matmul(lhs, rhs, arr, *ctx);
            }
            else
            {
                strided_matmul(lhs, rhs, arr, *ctx);
            }
        }
        else
        {
            if (binary_op->is_in_place())
            {
                // Share memory with lhs
                arr->alloc(*lhs->get_buff());
            }
            else
            {
                arr->alloc();
            }
            if (lhs->is_contiguous() && rhs->is_contiguous())
            {
                binary_ss(binary_op->get_name_str(), lhs, rhs, arr, *ctx);
            }
            else
            {
                strided_binary_ss(binary_op->get_name_str(), lhs, rhs, arr, *ctx);
            }
        }
    }

    void MTLGraph::call_transform(ArrayPtr arr)
    {
        auto op = arr->get_op();
        switch (op->get_name())
        {
        case OpName::RESHAPE:
        {
            auto reshape_op = std::static_pointer_cast<ReshapeOp>(op);
            auto operand = reshape_op->get_operand();
            if (operand->is_contiguous())
            {
                arr->alloc(*operand->get_buff());
            }
            else
            {
                arr->alloc();
                strided_copy(operand, arr, *ctx);
            }
            break;
        }
        case OpName::SLICE:
        {
            auto slice_op = std::static_pointer_cast<SliceOp>(op);
            auto operand = slice_op->get_operand();
            arr->alloc(*operand->get_buff());
            break;
        }
        case OpName::BROADCAST:
        {
            auto broadcast_op = std::static_pointer_cast<BroadcastOp>(op);
            auto operand = broadcast_op->get_operand();
            arr->alloc(*operand->get_buff());
            break;
        }
        case OpName::PERMUTE:
        {
            auto permute_op = std::static_pointer_cast<PermuteOp>(op);
            auto operand = permute_op->get_operand();
            arr->alloc(*operand->get_buff());
            break;
        }
        default:
            break;
        }
    }

    void MTLGraph::call_reduce(ArrayPtr arr)
    {
        auto op = arr->get_op();
        auto reduce_op = std::static_pointer_cast<ReduceOp>(op);
        auto operand = reduce_op->get_operand();
        arr->alloc();
        reduce(reduce_op->get_name_str(), operand, arr, *ctx);
    }

    void MTLGraph::call_move(ArrayPtr arr)
    {
        auto move_op = std::static_pointer_cast<TransformOp>(arr->get_op());
        auto operand = move_op->get_operand();
        arr->alloc();
        if (arr->is_contiguous() && operand->is_contiguous())
        {
            copy(operand, arr, *ctx);
        }
        else
        {
            strided_copy(operand, arr, *ctx);
        }
    }

    void MTLGraph::toposort(ArrayPtr arr, std::vector<ArrayPtr> &order)
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
        case OpType::TRANSFORM:
        {
            auto transform_op = std::static_pointer_cast<TransformOp>(arr->get_op());
            auto operand = transform_op->get_operand();
            toposort(operand, order);
            order.push_back(arr);
            break;
        }
        case OpType::REDUCE:
        {
            auto reduce_op = std::static_pointer_cast<ReduceOp>(arr->get_op());
            auto operand = reduce_op->get_operand();
            toposort(operand, order);
            order.push_back(arr);
            break;
        }
        default:
            // Move operations
            auto move_op = std::static_pointer_cast<CopyOp>(arr->get_op());
            auto operand = move_op->get_operand();
            toposort(operand, order);
            order.push_back(arr);
            break;
        }
    }

    void MTLGraph::call(ArrayPtr arr)
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
            call_unary(arr);
            break;
        }
        case OpType::BINARY:
        {
            call_binary(arr);
            break;
        }
        case OpType::TRANSFORM:
        {
            call_transform(arr);
            break;
        }
        case OpType::REDUCE:
        {
            call_reduce(arr);
            break;
        }
        default:
        {
            call_move(arr);
            break;
        }
        }
    }

    void MTLGraph::compile()
    {
        if (fw_order.empty())
        {
            // if (root->get_numel() > 1)
            // {
            //     throw std::invalid_argument("Root array " + root->get_id().str() + " must contain a single element.");
            // }
            toposort(root, fw_order);
            // Seed root for now
            root->grad = Array::ones_like(root, root->get_device());
            // Initializes the gradient array first without allocating buffers
            for (auto &arr : std::views::reverse(fw_order))
            {
                arr->get_op()->backward(arr);
            }
            // Order the gradient arrays
            for (auto &arr : std::views::reverse(fw_order))
            {
                // grad is null when backward is not implemented for op
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
            if (arr->get_op()->get_type() != OpType::INITIALIZER || (arr->get_op()->get_type() == OpType::INITIALIZER && arr->get_buff() == nullptr))
            {
                // Call initializers only once
                call(arr);
            }
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
            s += arr->get_id().str() + ": " + arr->get_op()->str() + "\n";
        }
        s += "Backward:\n";
        for (auto &arr : bw_order)
        {
            s += arr->get_id().str() + ": " + arr->get_op()->str() + "\n";
        }
        return s;
    }
}