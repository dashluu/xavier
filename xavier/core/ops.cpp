#include "array.h"
#include "ops.h"

namespace xv::core
{
    const std::string UnaryOp::str() const
    {
        return get_name_str() + ", operand: " + std::to_string(operand->get_id());
    }

    const std::string RootBinaryOp::str() const
    {
        return get_name_str() + ", lhs: " + std::to_string(lhs->get_id()) + ", rhs: " + std::to_string(rhs->get_id());
    }

    const std::string TransformOp::str() const
    {
        return get_name_str() + ", operand: " + std::to_string(operand->get_id());
    }

    const std::string MoveOp::str() const
    {
        return get_name_str() + ", operand: " + std::to_string(operand->get_id());
    }

    void AddOp::backward(std::shared_ptr<Array> arr) const
    {
        lhs->init_grad();
        rhs->init_grad();
        lhs->grad = lhs->grad->self_add(arr->grad);
        rhs->grad = rhs->grad->self_add(arr->grad);
    }

    void SelfAddOp::backward(std::shared_ptr<Array> arr) const
    {
        // x += y
        // dy += dx
        lhs->grad = arr->grad;
        rhs->init_grad();
        rhs->grad = rhs->grad->self_add(arr->grad);
    }

    void SubOp::backward(std::shared_ptr<Array> arr) const
    {
        lhs->init_grad();
        rhs->init_grad();
        lhs->grad = lhs->grad->self_add(arr->grad);
        rhs->grad = rhs->grad->self_sub(arr->grad);
    }

    void MulOp::backward(std::shared_ptr<Array> arr) const
    {
        // z = x*y
        // dx += dz*y
        // dy += dz*x
        lhs->init_grad();
        rhs->init_grad();
        lhs->grad = lhs->grad->self_add(arr->grad->mul(rhs));
        rhs->grad = rhs->grad->self_add(arr->grad->mul(lhs));
    }

    void DivOp::backward(std::shared_ptr<Array> arr) const
    {
        // z = x/y
        // dx += dz * (1/y)
        // dy += dz * (-x / y**2)
        lhs->init_grad();
        rhs->init_grad();
        lhs->grad = lhs->grad->self_add(arr->grad->div(rhs));
        rhs->grad = rhs->grad->self_sub(arr->grad->mul(arr->div(rhs)));
    }

    void SqOp::backward(std::shared_ptr<Array> arr) const
    {
        // z = x**2
        // dx += dz * 2x
        auto grad = arr->grad;
        // backward_helper(arr, grad->mul(operand->mul(2.0f)));
    }

    void ExpOp::backward(std::shared_ptr<Array> arr) const
    {
        // z = e**x
        // dx += dz * e**x
        operand->init_grad();
        operand->grad = operand->grad->self_add(arr->grad->mul(arr));
    }

    void LogOp::backward(std::shared_ptr<Array> arr) const
    {
        // z = ln(x)
        // dx += dz / x
        operand->init_grad();
        operand->grad = operand->grad->self_add(arr->grad->div(operand));
    }

    void NegOp::backward(std::shared_ptr<Array> arr) const
    {
        // z = -x
        // dx += -dz
        operand->init_grad();
        operand->grad = operand->grad->self_sub(arr->grad);
    }

    void RecipOp::backward(std::shared_ptr<Array> arr) const
    {
        // z = 1/x
        // dx += dz * -1/x**2
        operand->init_grad();
        operand->grad = operand->grad->self_sub(arr->grad->mul(arr->sq()));
    }
}