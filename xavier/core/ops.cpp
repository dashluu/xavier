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

    const std::string TransformOp::str() const
    {
        return get_name_str() + ", operand: " + operand->get_id().str();
    }

    const std::string ReduceOp::str() const
    {
        return get_name_str() + ", operand: " + operand->get_id().str();
    }

    const std::string CopyOp::str() const
    {
        return get_name_str() + ", operand: " + operand->get_id().str();
    }

    void AddOp::backward(std::shared_ptr<Array> arr) const
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

    void SubOp::backward(std::shared_ptr<Array> arr) const
    {
        // z = x + y
        // dx += dz
        // dy -= dz
        lhs->init_grad();
        lhs->update_grad(arr->grad);
        rhs->init_grad();
        rhs->update_grad(arr->grad, true);
    }

    void MulOp::backward(std::shared_ptr<Array> arr) const
    {
        // z = x*y
        // dx += dz*y
        // dy += dz*x
        lhs->init_grad();
        lhs->update_grad(arr->grad->mul(rhs));
        rhs->init_grad();
        rhs->update_grad(arr->grad->mul(lhs));
    }

    void DivOp::backward(std::shared_ptr<Array> arr) const
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

    void MatmulOp::backward(std::shared_ptr<Array> arr) const
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

    void SqOp::backward(std::shared_ptr<Array> arr) const
    {
        // z = x**2
        // dx += dz * 2x
        operand->init_grad();
        operand->update_grad(arr->grad->mul(operand->mul(2)));
    }

    void SqrtOp::backward(std::shared_ptr<Array> arr) const
    {
        // z = sqrt(x)
        // dx += dz / (2 * sqrt(x))
        // dx += dz / 2z
        operand->init_grad();
        operand->update_grad(arr->grad->div(arr->mul(2)));
    }

    void ExpOp::backward(std::shared_ptr<Array> arr) const
    {
        // z = e**x
        // dx += dz * e**x
        // dx += dz * z
        operand->init_grad();
        operand->update_grad(arr->grad->mul(arr));
    }

    void LogOp::backward(std::shared_ptr<Array> arr) const
    {
        // z = ln(x)
        // dx += dz / x
        operand->init_grad();
        operand->update_grad(arr->grad->div(operand));
    }

    void NegOp::backward(std::shared_ptr<Array> arr) const
    {
        // z = -x
        // dx += -dz
        // dx -= dz
        operand->init_grad();
        operand->update_grad(arr->grad, true);
    }

    void RecipOp::backward(std::shared_ptr<Array> arr) const
    {
        // z = 1/x
        // dx += dz * -1/x**2
        // dx += dz * -z**2
        // dx -= dz * z**2
        operand->init_grad();
        operand->update_grad(arr->grad->mul(arr->sq()), true);
    }

    void ReshapeOp::backward(std::shared_ptr<Array> arr) const
    {
        // Copy is done to ensure gradient independence
        // Copy first and then reshape for efficiency
        operand->init_grad();
        operand->update_grad(arr->grad->copy()->reshape(operand->get_view()));
    }

    void SliceOp::backward(std::shared_ptr<Array> arr) const
    {
        // operand->init_grad();
        // operand->update_grad(arr->grad->unslice(operand->get_shape(), ranges));
    }

    void UnsliceOp::backward(std::shared_ptr<Array> arr) const
    {
        // operand->init_grad();
        // operand->update_grad(arr->grad->slice(ranges));
    }

    void PermuteOp::backward(std::shared_ptr<Array> arr) const
    {
        operand->init_grad();
        // Gradient independence
        auto grad_copy = arr->grad->copy();
        auto reversed_order = grad_copy->get_shape().undo_permute_view(order);
        operand->update_grad(grad_copy->permute(reversed_order));
    }
}