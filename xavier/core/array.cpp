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
                throw std::runtime_error("Invalid starting point for range: " + std::to_string(range.start));
            }
            if (range.stop > shape[i])
            {
                throw std::runtime_error("Invalid stopping point for range: " + std::to_string(range.stop));
            }
            if (range.step == 0)
            {
                throw std::runtime_error("Step cannot be zero.");
            }
            if (range.start < range.stop && range.step < 0)
            {
                throw std::runtime_error("Step must be positive if start < stop: " + std::to_string(range.start) + " < " + std::to_string(range.stop));
            }
            if (range.start > range.stop && range.step > 0)
            {
                throw std::runtime_error("Step must be negative if start > stop: " + std::to_string(range.start) + " > " + std::to_string(range.stop));
            }
        }
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
            auto ptr = iter->curr();
            s += fmt(ptr, dtype);
            for (int i = elms_per_dim.size() - 1; i >= 0; i--)
            {
                if (iter->count() % elms_per_dim[i] == 0)
                {
                    s += "]";
                    close += 1;
                }
            }
            iter->next();
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

    void Array::copy_to(std::shared_ptr<const Array> dest) const
    {
        auto nbytes1 = get_nbytes();
        auto nbytes2 = dest->get_nbytes();
        if (nbytes1 != nbytes2)
        {
            throw std::invalid_argument("Cannot copy arrays of different sizes: " +
                                        std::to_string(nbytes1) + " and " + std::to_string(nbytes2) + ".");
        }
        if (is_contiguous() && dest->is_contiguous())
        {
            std::copy_n(get_ptr(), get_nbytes(), dest->get_ptr());
        }
        else
        {
            auto iter1 = std::make_unique<ArrayIter>(shared_from_this());
            auto iter2 = std::make_unique<ArrayIter>(dest);
            auto itemsize = dtype.get_size();
            for (iter1->start(), iter2->start(); iter1->has_next(); iter1->next(), iter2->next())
            {
                std::memcpy(iter2->curr(), iter1->curr(), itemsize);
            }
        }
    }

    std::shared_ptr<Array> Array::interpret_(const Dtype &dtype)
    {
        if (!is_contiguous())
        {
            throw std::runtime_error("Cannot interpret non-contiguous array.");
        }
        if (shape.get_ndim() != 1)
        {
            throw std::runtime_error("Only arrays with one dimension can be interpreted, but got one with " +
                                     std::to_string(shape.get_ndim()) + " dimensions.");
        }
        double u = static_cast<double>(this->dtype.get_size()) / dtype.get_size();
        double v = static_cast<double>(shape.get_numel()) * u;
        uint64_t numel = static_cast<uint64_t>(v);
        auto arr = std::make_shared<Array>(get_ptr(), Shape(shape.get_offset(), {numel}, {1}), dtype, device);
        return arr;
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

    std::shared_ptr<Array> Array::add(std::shared_ptr<Array> rhs) { return binary<AddOp>(rhs, binary_dtypes); }

    std::shared_ptr<Array> Array::sub(std::shared_ptr<Array> rhs) { return binary<SubOp>(rhs, binary_dtypes); }

    std::shared_ptr<Array> Array::mul(std::shared_ptr<Array> rhs) { return binary<MulOp>(rhs, binary_dtypes); }

    std::shared_ptr<Array> Array::div(std::shared_ptr<Array> rhs) { return binary<DivOp>(rhs, binary_dtypes); }

    std::shared_ptr<Array> Array::reshape_(const std::vector<uint64_t> &view)
    {
        Shape::check_view(view);
        uint64_t numel = std::accumulate(view.begin(), view.end(), 1, std::multiplies<uint64_t>());
        if (shape.get_numel() != numel)
        {
            throw std::invalid_argument("Cannot reshape array of " + std::to_string(shape.get_numel()) +
                                        " to " + std::to_string(numel) + " elements.");
        }
        auto s = Shape(shape.get_offset(), view);
        if (is_contiguous())
        {
            return std::make_shared<Array>(get_ptr(), s, dtype, device);
        }
        auto arr = std::make_shared<Array>(s, dtype, device);
        copy_to(arr);
        return arr;
    }
}
