#include "array.h"
#include "ops.h"

namespace xv::core
{
    const std::string UnaryOp::str() const
    {
        return get_name_str() + ", in-place: " + std::to_string(in_place) + ", operand: " + operand->get_id().str();
    }

    const std::string BinaryOp::str() const
    {
        return get_name_str() + ", in-place: " + std::to_string(in_place) + ", lhs: " + lhs->get_id().str() + ", rhs: " + rhs->get_id().str();
    }

    const std::string MatmulOp::str() const
    {
        return get_name_str() + ", lhs: " + lhs->get_id().str() + ", rhs: " + rhs->get_id().str();
    }

    const std::string TransformOp::str() const
    {
        return get_name_str() + ", operand: " + operand->get_id().str();
    }

    const std::string ReduceOp::str() const
    {
        return get_name_str() + ", operand: " + operand->get_id().str() + ", dims: " + vnumstr(dims);
    }

    void AddOp::backward(ArrayPtr arr) const
    {
        // In-place or not, gradient should be computed properly
        // z = x + y
        // dx += dz
        // dy += dz
        lhs->init_grad();
        lhs->update_grad(arr->grad);
        rhs->init_grad();
        rhs->update_grad(arr->grad);
    }

    void SubOp::backward(ArrayPtr arr) const
    {
        // z = x + y
        // dx += dz
        // dy -= dz
        lhs->init_grad();
        lhs->update_grad(arr->grad);
        rhs->init_grad();
        rhs->update_grad(arr->grad, true);
    }

    void MulOp::backward(ArrayPtr arr) const
    {
        // z = x*y
        // dx += dz*y
        // dy += dz*x
        lhs->init_grad();
        lhs->update_grad(arr->grad->mul(rhs));
        rhs->init_grad();
        rhs->update_grad(arr->grad->mul(lhs));
    }

    void DivOp::backward(ArrayPtr arr) const
    {
        // z = x/y
        // dx += dz * (1/y)
        // dy += dz * (-x / y**2)
        // dy -= dz * (z / y)
        lhs->init_grad();
        lhs->update_grad(arr->grad->div(rhs));
        rhs->init_grad();
        rhs->update_grad(arr->grad->mul(arr->div(rhs)), true);
    }

    void MatmulOp::backward(ArrayPtr arr) const
    {
        // z = x@y
        // dx += dz @ y^T
        // dy += x^T @ dz
        lhs->init_grad();
        // Transpose the last two dimensions of lhs and rhs
        lhs->update_grad(arr->grad->matmul(rhs->T(rhs->get_ndim() - 2)));
        rhs->init_grad();
        rhs->update_grad(lhs->T(lhs->get_ndim() - 2)->matmul(arr->grad));
    }

    void SqOp::backward(ArrayPtr arr) const
    {
        // z = x**2
        // dx += dz * 2x
        operand->init_grad();
        operand->update_grad(arr->grad->mul(operand->mul(2.0f)));
    }

    void SqrtOp::backward(ArrayPtr arr) const
    {
        // z = sqrt(x)
        // dx += dz / (2 * sqrt(x))
        // dx += dz / 2z
        operand->init_grad();
        operand->update_grad(arr->grad->div(arr->mul(2.0f)));
    }

    void ExpOp::backward(ArrayPtr arr) const
    {
        // z = e**x
        // dx += dz * e**x
        // dx += dz * z
        operand->init_grad();
        operand->update_grad(arr->grad->mul(arr));
    }

    void LogOp::backward(ArrayPtr arr) const
    {
        // z = ln(x)
        // dx += dz / x
        operand->init_grad();
        operand->update_grad(arr->grad->div(operand));
    }

    void NegOp::backward(ArrayPtr arr) const
    {
        // z = -x
        // dx += -dz
        // dx -= dz
        operand->init_grad();
        operand->update_grad(arr->grad, true);
    }

    void IdOp::backward(ArrayPtr arr) const
    {
        // z = x
        // dx += dz
        operand->init_grad();
        operand->update_grad(arr->grad, false);
    }

    void RecipOp::backward(ArrayPtr arr) const
    {
        // z = 1/x
        // dx += dz * -1/x**2
        // dx += dz * -z**2
        // dx -= dz * z**2
        operand->init_grad();
        operand->update_grad(arr->grad->mul(arr->sq()), true);
    }

    void ReshapeOp::backward(ArrayPtr arr) const
    {
        operand->init_grad();
        // Copy must be done to ensure gradient independence
        if (arr->grad->copy_when_reshape())
        {
            operand->update_grad(arr->grad->reshape(operand->get_view()));
        }
        else
        {
            // Copy first and then reshape because reshaping to incompatible shape might cause another copy
            operand->update_grad(arr->grad->identity()->reshape(operand->get_view()));
        }
    }

    void SliceOp::backward(ArrayPtr arr) const
    {
        operand->init_grad();
        std::cout << vstr<Range>(ranges, [](Range range)
                                 { return range.str(); })
                  << std::endl;
        operand->grad_root = operand->grad->slice(ranges)->self_add(arr->grad);
    }

    void PermuteOp::backward(ArrayPtr arr) const
    {
        operand->init_grad();
        // Copy must be done to ensure gradient independence
        auto grad_copy = arr->grad->identity();
        auto reversed_order = grad_copy->get_shape().undo_permute_view(order);
        operand->update_grad(grad_copy->permute(reversed_order));
    }

    void SumOp::backward(ArrayPtr arr) const
    {
        operand->init_grad();
        operand->update_grad(arr->grad);
    }
}