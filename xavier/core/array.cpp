#include "iter.h"
#include "array.h"

namespace xv::core
{
    std::string Array::fmt(uint8_t *ptr, const Dtype &dtype) const
    {
        switch (dtype.get_size())
        {
        case 1:
        {
            int8_t val = *reinterpret_cast<int8_t *>(ptr);
            return dtype == i8 ? std::to_string(val) : (val ? "True" : "False");
        }
        case 2:
            return std::to_string(*reinterpret_cast<int16_t *>(ptr));
        case 4:
            return fmt_num<int32_t, float>(ptr, dtype);
        default:
            throw std::invalid_argument("Unsupported dtype.");
        }
    }

    void Array::check_dims(usize start_dim, usize end_dim) const
    {
        if (start_dim > end_dim)
        {
            throw std::invalid_argument("The start dimension must be smaller than the end dimension.");
        }
        if (start_dim >= get_ndim())
        {
            throw std::invalid_argument("The start dimension must be smaller than the number of dimensions.");
        }
        if (end_dim >= get_ndim())
        {
            throw std::invalid_argument("The end dimension must be smaller than the number of dimensions.");
        }
    }

    uint8_t *Array::strided_idx(usize k) const
    {
        if (is_contiguous())
        {
            return get_ptr() + k * get_itemsize();
        }
        std::vector<usize> idx(get_ndim());
        usize carry = k;
        usize tmp;
        for (int i = get_ndim() - 1; i >= 0; i--)
        {
            tmp = carry;
            idx[i] = tmp % shape[i];
            carry = tmp / shape[i];
        }
        auto &stride = get_stride();
        auto ptr = get_ptr();
        for (size_t i = 0; i < idx.size(); i++)
        {
            ptr += idx[i] * stride[i] * get_itemsize();
        }
        return ptr;
    }

    const std::string Array::str() const
    {
        auto iter = std::make_unique<ArrayIter>(shared_from_this());
        iter->start();
        bool flag = iter->has_next();
        std::string s = "";
        for (int i = 0; i < get_ndim(); i++)
        {
            s += "[";
        }
        if (!flag)
        {
            for (int i = 0; i < get_ndim(); i++)
            {
                s += "]";
            }
            return s;
        }
        auto elms_per_dim = shape.get_elms_per_dim();
        int close = 0;
        while (flag)
        {
            close = 0;
            auto ptr = iter->next();
            // std::cout << std::hex << static_cast<void *>(ptr) << std::endl;
            s += fmt(ptr, dtype);
            for (int i = elms_per_dim.size() - 1; i >= 0; i--)
            {
                if (iter->count() % elms_per_dim[i] == 0)
                {
                    s += "]";
                    close += 1;
                }
            }
            flag = iter->has_next();
            if (flag)
            {
                if (close > 0)
                {
                    s += ", \n";
                }
                else
                {
                    s += ", ";
                }
                for (int i = 0; i < close; i++)
                {
                    s += "[";
                }
            }
        }
        return s;
    }

    ArrayPtr Array::slice(const std::vector<Range> &ranges)
    {
        auto arr = std::make_shared<Array>(shape.slice(ranges), dtype, device);
        arr->op = std::make_shared<SliceOp>(shared_from_this(), ranges);
        return arr;
    }

    ArrayPtr Array::arange(const ShapeView &view, isize start, isize step, const Dtype &dtype, const Device &device, bool constant)
    {
        auto op = std::make_shared<ArangeOp>(view, start, step, dtype);
        auto arr = std::make_shared<Array>(Shape(view), dtype, device, constant);
        arr->op = op;
        return arr;
    }

    ArrayPtr Array::full(const ShapeView &view, int c, const Dtype &dtype, const Device &device, bool constant)
    {
        std::shared_ptr<Op> op;
        if (dtype == f32)
        {
            op = std::make_shared<FullOp>(view, std::bit_cast<int>(static_cast<float>(c)), dtype);
        }
        else
        {
            op = std::make_shared<FullOp>(view, c, dtype);
        }
        auto arr = std::make_shared<Array>(Shape(view), dtype, device, constant);
        arr->op = op;
        return arr;
    }

    ArrayPtr Array::full(const ShapeView &view, float c, const Dtype &dtype, const Device &device, bool constant)
    {
        std::shared_ptr<Op> op;
        if (dtype == f32)
        {
            op = std::make_shared<FullOp>(view, std::bit_cast<int>(c), dtype);
        }
        else
        {
            op = std::make_shared<FullOp>(view, static_cast<int>(c), dtype);
        }
        auto arr = std::make_shared<Array>(Shape(view), dtype, device, constant);
        arr->op = op;
        return arr;
    }

    ArrayPtr Array::from_buff(uint8_t *ptr, usize nbytes, const Shape &shape, const Dtype &dtype, const Device &device, bool constant)
    {
        auto op = std::make_shared<BuffOp>();
        auto arr = std::make_shared<Array>(ptr, nbytes, shape, dtype, device, constant);
        arr->op = op;
        return arr;
    }

    ArrayPtr Array::from_numpy(uint8_t *ptr, usize nbytes, const Shape &shape, const Dtype &dtype, const Device &device, bool constant)
    {
        auto op = std::make_shared<NumpyOp>();
        auto arr = std::make_shared<Array>(ptr, nbytes, shape, dtype, device, constant);
        arr->op = op;
        return arr;
    }

    ArrayPtr Array::matmul(ArrayPtr rhs)
    {
        auto dummy_op = std::make_shared<MatmulOp>(nullptr, nullptr);
        auto &rview = rhs->get_view();
        if (!shape.matmul_broadcastable(rview))
        {
            throw IncompatShapesForOp(dummy_op->get_name_str(), vnumstr(get_view()), vnumstr(rview));
        }
        if (!binary_dtypes.contains(dtype) || dtype != rhs->dtype)
        {
            throw IncompatDtypesForOp(dummy_op->get_name_str(), dtype.str(), rhs->dtype.str());
        }
        if (device != rhs->get_device())
        {
            throw IncompatDevicesForOp(dummy_op->get_name_str(), device.str(), rhs->device.str());
        }
        auto broadcasted_lview = get_view();
        auto broadcasted_rview = rview;
        auto ndim = std::max(broadcasted_lview.size(), broadcasted_rview.size());
        broadcasted_lview.insert(broadcasted_lview.begin(), ndim - broadcasted_lview.size(), 1);
        broadcasted_rview.insert(broadcasted_rview.begin(), ndim - broadcasted_rview.size(), 1);
        for (int i = 0; i < ndim - 2; i++)
        {
            auto shared_dim = std::max(broadcasted_lview[i], broadcasted_rview[i]);
            broadcasted_lview[i] = shared_dim;
            broadcasted_rview[i] = shared_dim;
        }
        auto batch = std::accumulate(
            broadcasted_lview.begin(),
            std::prev(broadcasted_lview.end(), 2),
            1ULL,
            std::multiplies<usize>());
        ShapeView mm_lview = {
            batch,
            broadcasted_lview[broadcasted_lview.size() - 2],
            broadcasted_lview[broadcasted_lview.size() - 1]};
        ShapeView mm_rview = {
            batch,
            broadcasted_rview[broadcasted_rview.size() - 2],
            broadcasted_rview[broadcasted_rview.size() - 1]};
        // Broadcast lhs and rhs to have only 3D
        // Lhs's shape: B, M, N
        auto mm_lhs = broadcast(broadcasted_lview)->reshape(mm_lview);
        // Rhs's shape: B, N, K
        auto mm_rhs = rhs->broadcast(broadcasted_rview)->reshape(mm_rview);
        // Result's shape: B, M, K
        auto view = mm_lhs->get_view();
        view[view.size() - 1] = rview[rview.size() - 1];
        auto arr = std::make_shared<Array>(Shape(view), dtype, device);
        arr->op = std::make_shared<MatmulOp>(mm_lhs, mm_rhs);
        // Reshape to expected result's shape
        auto reshaped_view = broadcasted_lview;
        reshaped_view[reshaped_view.size() - 1] = rview[rview.size() - 1];
        arr = arr->reshape(reshaped_view);
        return arr;
    }

    ArrayPtr Array::reshape(const ShapeView &view)
    {
        if (get_view() == view)
        {
            return shared_from_this();
        }
        Shape::check_view(view);
        usize numel = std::accumulate(view.begin(), view.end(), 1, std::multiplies<usize>());
        if (shape.get_numel() != numel)
        {
            throw std::invalid_argument("Cannot reshape array of " + std::to_string(shape.get_numel()) +
                                        " to " + std::to_string(numel) + " elements.");
        }
        auto arr = std::make_shared<Array>(shape.reshape(view), dtype, device);
        arr->op = std::make_shared<ReshapeOp>(shared_from_this(), view);
        return arr;
    }

    ArrayPtr Array::broadcast(const ShapeView &view)
    {
        if (get_view() == view)
        {
            return shared_from_this();
        }
        auto arr = std::make_shared<Array>(shape.broadcast(view), dtype, device);
        arr->op = std::make_shared<BroadcastOp>(shared_from_this(), view);
        return arr;
    }

    ArrayPtr Array::broadcast_to(const ShapeView &view)
    {
        if (get_view() == view)
        {
            return shared_from_this();
        }
        auto arr = std::make_shared<Array>(shape.broadcast_to(view), dtype, device);
        arr->op = std::make_shared<BroadcastOp>(shared_from_this(), view);
        return arr;
    }

    ArrayPtr Array::identity()
    {
        auto arr = std::make_shared<Array>(Shape(get_view()), dtype, device);
        arr->op = std::make_shared<IdentityOp>(shared_from_this());
        return arr;
    }

    ArrayPtr Array::permute(const ShapeOrder &order)
    {
        auto arr = std::make_shared<Array>(shape.permute(order), dtype, device);
        arr->op = std::make_shared<PermuteOp>(shared_from_this(), order);
        return arr;
    }

    ArrayPtr Array::T(usize start_dim, usize end_dim)
    {
        check_dims(start_dim, end_dim);
        ShapeOrder order(get_ndim());
        std::iota(order.begin(), order.begin() + start_dim, 0);
        std::iota(order.begin() + end_dim + 1, order.end(), end_dim + 1);
        // Fill with decreasing values
        std::generate(order.begin() + start_dim, order.begin() + end_dim + 1, [n = end_dim]() mutable
                      { return n--; });
        return permute(order);
    }

    ArrayPtr Array::flatten(usize start_dim, usize end_dim)
    {
        check_dims(start_dim, end_dim);
        auto view = get_view();
        auto prod = std::accumulate(view.begin() + start_dim, view.begin() + end_dim + 1, 1ULL, std::multiplies<usize>());
        // Erase from start_dim + 1 to end_dim + 1
        view.erase(view.begin() + start_dim + 1, view.begin() + end_dim + 1);
        // Update view at start_dim
        view[start_dim] = prod;
        return reshape(view);
    }
}
