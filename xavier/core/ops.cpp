#include "array.h"
#include "ops.h"

namespace xv::core
{
    const std::string UnaryOp::str() const
    {
        return opnames.at(name) + ", operand: " + std::to_string(operand->get_id());
    }

    const std::string RootBinaryOp::str() const
    {
        return opnames.at(name) + ", lhs: " + std::to_string(lhs->get_id()) + ", rhs: " + std::to_string(rhs->get_id());
    }

    const std::string TransformOp::str() const
    {
        return opnames.at(name) + ", operand: " + std::to_string(operand->get_id());
    }

    void BinaryOp::backward_helper(std::shared_ptr<Array> arr, std::shared_ptr<Array> lgrad, std::shared_ptr<Array> rgrad) const
    {
        if (lhs->get_op()->get_type() != OpType::INITIALIZER)
        {
            if (lhs->grad == nullptr)
            {
                lhs->grad = Array::zeros_like(lhs);
            }
            lhs->grad = lhs->grad->iadd(lgrad);
        }
        if (rhs->get_op()->get_type() != OpType::INITIALIZER)
        {
            if (rhs->grad == nullptr)
            {
                rhs->grad = Array::zeros_like(rhs);
            }
            rhs->grad = rhs->grad->iadd(rgrad);
        }
    }

    void AddOp::backward(std::shared_ptr<Array> arr) const
    {
        auto grad = arr->grad;
        backward_helper(arr, grad, grad);
    }

    void SubOp::backward(std::shared_ptr<Array> arr) const
    {
        auto grad = arr->grad;
        backward_helper(arr, grad, grad->neg());
    }

    void MulOp::backward(std::shared_ptr<Array> arr) const
    {
        // z = x*y
        // dx += dz*y
        // dy += dz*x
        auto grad = arr->grad;
        backward_helper(arr, grad->mul(rhs), grad->mul(lhs));
    }

    void DivOp::backward(std::shared_ptr<Array> arr) const
    {
        // z = x/y
        // dx += dz * (1/y)
        // dy += dz * (-x / y^2)
        auto grad = arr->grad;
        backward_helper(arr, grad->mul(rhs->recip()), grad->mul(lhs->div(rhs->sq()))->neg());
    }
}