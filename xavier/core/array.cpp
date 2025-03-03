#include "iter.h"
#include "array.h"

namespace xv::core
{
    template <class I, class F>
    std::string fmt_num(uint8_t *ptr, Dtype dtype)
    {
        if (dtype.get_name().at(0) == 'i')
        {
            return std::to_string(*reinterpret_cast<I *>(ptr));
        }
        F val = *reinterpret_cast<F *>(ptr);
        if (0 < val && val <= 1e-5)
        {
            return std::format("{:.4e}", val);
        }
        return std::format("{:.4f}", val);
    }

    std::string fmt(uint8_t *ptr, const Dtype &dtype)
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
        case 8:
            return fmt_num<int64_t, double>(ptr, dtype);
        default:
            throw std::invalid_argument("Unsupported dtype.");
        }
    }

    void Array::check_ranges(const std::vector<Range> &ranges) const
    {
        if (ranges.size() != shape.get_ndim())
        {
            throw std::invalid_argument("The number of ranges does not match the number of dimensions: " +
                                        std::to_string(ranges.size()) + " and " +
                                        std::to_string(shape.get_ndim()) + ".");
        }
        for (int i = 0; i < ranges.size(); i++)
        {
            auto &range = ranges[i];
            if (range.start >= shape[i])
            {
                throw std::invalid_argument("Invalid starting point for range: " + std::to_string(range.start));
            }
            if (range.stop > shape[i])
            {
                throw std::invalid_argument("Invalid stopping point for range: " + std::to_string(range.stop));
            }
            if (range.step == 0)
            {
                throw std::invalid_argument("Step cannot be zero.");
            }
            if (range.start < range.stop && range.step < 0)
            {
                throw std::invalid_argument("Step must be positive if start < stop: " + std::to_string(range.start) + " < " + std::to_string(range.stop));
            }
            if (range.start > range.stop && range.step > 0)
            {
                throw std::invalid_argument("Step must be negative if start > stop: " + std::to_string(range.start) + " > " + std::to_string(range.stop));
            }
        }
    }

    uint8_t *Array::access_(uint64_t k) const
    {
        if (is_contiguous())
        {
            return get_ptr() + k * get_itemsize();
        }
        std::vector<uint64_t> idx(shape.get_ndim());
        uint carry = k;
        uint tmp;
        for (int i = shape.get_ndim() - 1; i >= 0; i--)
        {
            tmp = carry;
            idx[i] = tmp % shape[i];
            carry = tmp / shape[i];
        }
        auto stride = shape.get_stride();
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
        for (int i = 0; i < shape.get_ndim(); i++)
        {
            s += "[";
        }
        if (!flag)
        {
            for (int i = 0; i < shape.get_ndim(); i++)
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

    std::shared_ptr<Array> Array::slice(const std::vector<Range> &ranges)
    {
        check_ranges(ranges);
        const auto &stride1 = shape.get_stride();
        auto offset = shape.get_offset();
        for (int i = 0; i < ranges.size(); i++)
        {
            offset += ranges[i].start * stride1[i];
        }
        std::vector<uint64_t> view(shape.get_ndim());
        std::vector<int64_t> stride2(shape.get_ndim());
        for (int i = 0; i < ranges.size(); i++)
        {
            auto &range = ranges[i];
            auto d = range.start <= range.stop ? range.stop - range.start : range.start - range.stop;
            view[i] = static_cast<uint64_t>(ceil(static_cast<double>(d) / std::abs(range.step)));
            stride2[i] = stride1[i] * range.step;
        }
        auto arr = std::make_shared<Array>(Shape(offset, view, stride2), dtype, device);
        arr->op = std::make_shared<SliceOp>(shared_from_this(), ranges);
        return arr;
    }

    template <class O>
    std::shared_ptr<Array> Array::binary(std::shared_ptr<Array> rhs, const std::unordered_map<Dtype, Dtype> &dtype_map)
    {
        auto dummy_op = std::make_shared<O>(nullptr, nullptr);
        auto &rhs_shape = rhs->shape;
        auto &rhs_view = rhs_shape.get_view();
        if (!shape.broadcastable(rhs_view))
        {
            throw std::runtime_error("Cannot run operator " + dummy_op->str() + " on incompatible shapes " + numstr(shape.get_view()) + " and " + numstr(rhs_view) + ".");
        }
        if (dtype_map.find(dtype) == dtype_map.end() || dtype != rhs->dtype)
        {
            throw std::runtime_error("Cannot run operator " + dummy_op->str() + " on incompatible data types " + dtype.str() + " and " + rhs->dtype.str() + ".");
        }
        if (device != rhs->get_device())
        {
            throw std::runtime_error("Cannot run operator " + dummy_op->str() + " on incompatible devices " + device.str() + " and " + rhs->device.str() + ".");
        }
        auto broadcasted_shapes = shape.broadcast(rhs_shape);
        std::shared_ptr<Array> broadcasted_lhs;
        std::shared_ptr<Array> broadcasted_rhs;
        // If the lhs is already broadcasted, then we don't need to create a new array
        if (broadcasted_shapes.first == shape)
        {
            broadcasted_lhs = shared_from_this();
        }
        else
        {
            broadcasted_lhs = std::make_shared<Array>(broadcasted_shapes.first, dtype, device);
            broadcasted_lhs->op = std::make_shared<BroadcastOp>(shared_from_this(), broadcasted_lhs->shape.get_view());
        }
        // If the rhs is already broadcasted, then we don't need to create a new array
        if (broadcasted_shapes.second == rhs_shape)
        {
            broadcasted_rhs = rhs;
        }
        else
        {
            broadcasted_rhs = std::make_shared<Array>(broadcasted_shapes.second, dtype, device);
            broadcasted_rhs->op = std::make_shared<BroadcastOp>(rhs, broadcasted_rhs->shape.get_view());
        }
        auto result = std::make_shared<Array>(Shape(broadcasted_lhs->shape.get_view()), dtype, device);
        result->op = std::make_shared<O>(broadcasted_lhs, broadcasted_rhs);
        return result;
    }

    template <class O>
    std::shared_ptr<Array> Array::self_binary(std::shared_ptr<Array> rhs, const std::unordered_map<Dtype, Dtype> &dtype_map)
    {
        auto dummy_op = std::make_shared<O>(nullptr, nullptr);
        auto &rhs_shape = rhs->shape;
        auto &rhs_view = rhs_shape.get_view();
        if (!rhs_shape.broadcastable_to(shape.get_view()))
        {
            throw std::runtime_error("Cannot run operator " + dummy_op->str() + " on incompatible shapes " + numstr(shape.get_view()) + " and " + numstr(rhs_view) + ".");
        }
        if (dtype_map.find(dtype) == dtype_map.end() || dtype != rhs->dtype)
        {
            throw std::runtime_error("Cannot run operator " + dummy_op->str() + " on incompatible data types " + dtype.str() + " and " + rhs->dtype.str() + ".");
        }
        if (device != rhs->get_device())
        {
            throw std::runtime_error("Cannot run operator " + dummy_op->str() + " on incompatible devices " + device.str() + " and " + rhs->device.str() + ".");
        }
        auto broadcasted_shape = rhs_shape.broadcast_to(shape.get_view());
        std::shared_ptr<Array> broadcasted_rhs;
        // If the rhs is already broadcasted, then we don't need to create a new array
        if (broadcasted_shape == rhs_shape)
        {
            broadcasted_rhs = rhs;
        }
        else
        {
            broadcasted_rhs = std::make_shared<Array>(broadcasted_shape, dtype, device);
            broadcasted_rhs->op = std::make_shared<BroadcastOp>(rhs, broadcasted_rhs->shape.get_view());
        }
        auto result = std::make_shared<Array>(shape, dtype, device);
        result->op = std::make_shared<O>(shared_from_this(), broadcasted_rhs);
        return result;
    }

    template <class O>
    std::shared_ptr<Array> Array::unary(std::shared_ptr<Array> operand, const std::unordered_map<Dtype, Dtype> &dtype_map)
    {
        auto dummy_op = std::make_shared<O>(nullptr);
        if (dtype_map.find(dtype) == dtype_map.end())
        {
            throw std::runtime_error("Cannot run operator " + dummy_op->str() + " on incompatible data type " + dtype.str() + ".");
        }
        auto result = std::make_shared<Array>(Shape(shape.get_view()), dtype, device);
        result->op = std::make_shared<O>(shared_from_this());
        return result;
    }

    std::shared_ptr<Array> Array::add(std::shared_ptr<Array> rhs) { return binary<AddOp>(rhs, binary_dtypes); }

    std::shared_ptr<Array> Array::self_add(std::shared_ptr<Array> rhs) { return self_binary<SelfAddOp>(rhs, binary_dtypes); }

    std::shared_ptr<Array> Array::sub(std::shared_ptr<Array> rhs) { return binary<SubOp>(rhs, binary_dtypes); }

    std::shared_ptr<Array> Array::self_sub(std::shared_ptr<Array> rhs) { return self_binary<SelfSubOp>(rhs, binary_dtypes); }

    std::shared_ptr<Array> Array::mul(std::shared_ptr<Array> rhs) { return binary<MulOp>(rhs, binary_dtypes); }

    std::shared_ptr<Array> Array::self_mul(std::shared_ptr<Array> rhs) { return self_binary<SelfMulOp>(rhs, binary_dtypes); }

    std::shared_ptr<Array> Array::div(std::shared_ptr<Array> rhs) { return binary<DivOp>(rhs, binary_dtypes); }

    std::shared_ptr<Array> Array::self_div(std::shared_ptr<Array> rhs) { return self_binary<SelfDivOp>(rhs, binary_dtypes); }

    std::shared_ptr<Array> Array::sq() { return unary<SqOp>(shared_from_this(), unary_dtypes); }

    std::shared_ptr<Array> Array::sqrt() { return unary<SqrtOp>(shared_from_this(), unary_float_dtypes); }

    std::shared_ptr<Array> Array::exp() { return unary<ExpOp>(shared_from_this(), unary_float_dtypes); }

    std::shared_ptr<Array> Array::log() { return unary<LogOp>(shared_from_this(), unary_float_dtypes); }

    std::shared_ptr<Array> Array::neg() { return unary<NegOp>(shared_from_this(), unary_dtypes); }

    std::shared_ptr<Array> Array::recip() { return unary<RecipOp>(shared_from_this(), unary_float_dtypes); }

    std::shared_ptr<Array> Array::reshape(const std::vector<uint64_t> &view)
    {
        Shape::check_view(view);
        uint64_t numel = std::accumulate(view.begin(), view.end(), 1, std::multiplies<uint64_t>());
        if (shape.get_numel() != numel)
        {
            throw std::invalid_argument("Cannot reshape array of " + std::to_string(shape.get_numel()) +
                                        " to " + std::to_string(numel) + " elements.");
        }
        auto s = Shape(shape.get_offset(), view);
        std::shared_ptr<Array> arr;
        if (is_contiguous())
        {
            arr = std::make_shared<Array>(get_ptr(), s, dtype, device);
            auto op = std::make_shared<ReshapeOp>(shared_from_this(), view, false);
            arr->op = op;
        }
        else
        {
            // Copy is done on the GPU
            arr = std::make_shared<Array>(s, dtype, device);
            auto op = std::make_shared<ReshapeOp>(shared_from_this(), view, true);
            arr->op = op;
        }
        return arr;
    }
}
