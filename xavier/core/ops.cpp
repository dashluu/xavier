#include "array.h"
#include "ops.h"

namespace xv::core
{
    const std::string UnaryOp::str() const
    {
        return get_name_str() + ", in-place: " + std::to_string(in_place) + ", operand: " + std::to_string(operand->get_id());
    }

    const std::string BinaryOp::str() const
    {
        return get_name_str() + ", in-place: " + std::to_string(in_place) + ", lhs: " + std::to_string(lhs->get_id()) + ", rhs: " + std::to_string(rhs->get_id());
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
        if (in_place)
        {
            // x += y
            // dy += dx
            lhs->cum_grad = arr->cum_grad;
        }
        else
        {
            // z = x + y
            // dx += dz
            // dy += dz
            lhs->init_grad();
            lhs->grad = arr->grad;
            lhs->update_grad();
        }
        rhs->init_grad();
        rhs->grad = arr->grad;
        rhs->update_grad();
    }

    void SubOp::backward(std::shared_ptr<Array> arr) const
    {
        if (in_place)
        {
            // x -= y
            // dy -= dx
            lhs->cum_grad = arr->cum_grad;
        }
        else
        {
            // z = x + y
            // dx += dz
            // dy -= dz
            lhs->init_grad();
            lhs->grad = arr->grad;
            lhs->update_grad();
        }
        rhs->init_grad();
        rhs->grad = arr->grad->neg();
        rhs->update_grad();
    }

    void MulOp::backward(std::shared_ptr<Array> arr) const
    {
        if (in_place)
        {
            // x *= y
            // dy += dx*x
            lhs->cum_grad = arr->cum_grad;
        }
        else
        {
            // z = x*y
            // dx += dz*y
            // dy += dz*x
            lhs->init_grad();
            lhs->grad = arr->grad->mul(rhs);
            lhs->update_grad();
        }
        rhs->init_grad();
        rhs->grad = arr->grad->mul(lhs);
        rhs->update_grad();
    }

    void DivOp::backward(std::shared_ptr<Array> arr) const
    {
        if (in_place)
        {
            // x /= y
            // dy += dx * (-x / y**2)
            lhs->cum_grad = arr->cum_grad;
        }
        else
        {
            // z = x/y
            // dx += dz * (1/y)
            // dy += dz * (-x / y**2)
            lhs->init_grad();
            lhs->grad = arr->grad->div(rhs);
            lhs->update_grad();
        }
        rhs->init_grad();
        rhs->grad = arr->grad->mul(lhs->neg())->div(rhs->sq());
        rhs->update_grad();
    }

    void MatmulOp::backward(std::shared_ptr<Array> arr) const
    {
    }

    void SqOp::backward(std::shared_ptr<Array> arr) const
    {
        if (in_place)
        {
            // x **= 2
            operand->cum_grad = arr->cum_grad;
        }
        else
        {
            // z = x**2
            // dx += dz * 2x
            operand->init_grad();
            operand->grad = arr->grad->mul(operand->mul(2));
            operand->update_grad();
        }
    }

    void SqrtOp::backward(std::shared_ptr<Array> arr) const
    {
        if (in_place)
        {
            // x = sqrt(x)
            operand->cum_grad = arr->cum_grad;
        }
        else
        {
            // z = sqrt(x)
            // dx += dz / (2 * sqrt(x))
            // dx += dz / 2z
            operand->init_grad();
            operand->grad = arr->grad->div(arr->mul(2));
            operand->update_grad();
        }
    }

    void ExpOp::backward(std::shared_ptr<Array> arr) const
    {
        if (in_place)
        {
            // x = e**x
            operand->cum_grad = arr->cum_grad;
        }
        else
        {
            // z = e**x
            // dx += dz * e**x
            // dx += dz * z
            operand->init_grad();
            operand->grad = arr->grad->mul(arr);
            operand->update_grad();
        }
    }

    void LogOp::backward(std::shared_ptr<Array> arr) const
    {
        if (in_place)
        {
            // x = ln(x)
            operand->cum_grad = arr->cum_grad;
        }
        else
        {
            // z = ln(x)
            // dx += dz / x
            operand->init_grad();
            operand->grad = arr->grad->div(operand);
            operand->update_grad();
        }
    }

    void NegOp::backward(std::shared_ptr<Array> arr) const
    {
        if (in_place)
        {
            // x = -x
            operand->cum_grad = arr->cum_grad;
        }
        else
        {
            // z = -x
            // dx += -dz
            operand->init_grad();
            operand->grad = arr->grad->neg();
            operand->update_grad();
        }
    }

    void RecipOp::backward(std::shared_ptr<Array> arr) const
    {
        if (in_place)
        {
            // x = 1/x
            operand->cum_grad = arr->cum_grad;
        }
        else
        {
            // z = 1/x
            // dx += dz * -1/x**2
            // dx += dz * -z**2
            operand->init_grad();
            operand->grad = arr->grad->mul(arr->sq()->neg());
            operand->update_grad();
        }
    }
}