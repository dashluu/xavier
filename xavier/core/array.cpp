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
        default:
            throw std::invalid_argument("Unsupported dtype.");
        }
    }

    void Array::check_dims(uint64_t start_dim, uint64_t end_dim) const
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

    uint8_t *Array::strided_idx(uint64_t k) const
    {
        if (is_contiguous())
        {
            return get_ptr() + k * get_itemsize();
        }
        std::vector<uint64_t> idx(get_ndim());
        uint carry = k;
        uint tmp;
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

    std::shared_ptr<Array> Array::slice(const std::vector<Range> &ranges)
    {
        auto arr = std::make_shared<Array>(shape.slice(ranges), dtype, device);
        arr->op = std::make_shared<SliceOp>(shared_from_this(), ranges);
        return arr;
    }

    std::shared_ptr<Array> Array::unslice(const Shape &orig_shape, const std::vector<Range> &ranges)
    {
        Shape sliced_shape = orig_shape.slice(ranges);
        if (sliced_shape != shape)
        {
            throw std::invalid_argument("Cannot unslice because the sliced shape " + sliced_shape.str() + " does not match the current shape " + shape.str() + " of array " + id.str() + ".");
        }
        auto arr = std::make_shared<Array>(orig_shape, dtype, device, true);
        arr->op = std::make_shared<UnsliceOp>(shared_from_this(), orig_shape, ranges);
        return arr;
    }

    std::shared_ptr<Array> Array::arange(const std::vector<uint64_t> &view, int64_t start, int64_t step, const Dtype &dtype, const Device &device, bool constant)
    {
        auto op = std::make_shared<ArangeOp>(view, start, step, dtype);
        auto arr = std::make_shared<Array>(Shape(view), dtype, device, constant);
        arr->op = op;
        return arr;
    }

    std::shared_ptr<Array> Array::full(const std::vector<uint64_t> &view, int c, const Dtype &dtype, const Device &device, bool constant)
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

    std::shared_ptr<Array> Array::full(const std::vector<uint64_t> &view, float c, const Dtype &dtype, const Device &device, bool constant)
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

    template <class O>
    std::shared_ptr<Array> Array::binary_ss(std::shared_ptr<Array> rhs)
    {
        auto dummy_op = std::make_shared<O>(nullptr, nullptr, false);
        auto &rview = rhs->get_view();
        if (!shape.broadcastable(rview))
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
        auto broadcasted_lhs = broadcast(rview);
        auto broadcasted_rhs = rhs->broadcast(get_view());
        auto arr = std::make_shared<Array>(Shape(broadcasted_lhs->get_view()), dtype, device);
        arr->op = std::make_shared<O>(broadcasted_lhs, broadcasted_rhs, false);
        return arr;
    }

    template <class O>
    std::shared_ptr<Array> Array::self_binary_ss(std::shared_ptr<Array> rhs)
    {
        if (constant)
        {
            throw std::runtime_error("Cannot update array " + id.str() + " since it is a constant.");
        }
        auto dummy_op = std::make_shared<O>(nullptr, nullptr, true);
        if (!rhs->shape.broadcastable_to(get_view()))
        {
            throw IncompatShapesForOp(dummy_op->get_name_str(), vnumstr(get_view()), vnumstr(rhs->get_view()));
        }
        if (!binary_dtypes.contains(dtype) || dtype != rhs->dtype)
        {
            throw IncompatDtypesForOp(dummy_op->get_name_str(), dtype.str(), rhs->dtype.str());
        }
        if (device != rhs->get_device())
        {
            throw IncompatDevicesForOp(dummy_op->get_name_str(), device.str(), rhs->device.str());
        }
        auto broadcasted_rhs = rhs->broadcast_to(get_view());
        auto arr = std::make_shared<Array>(shape, dtype, device);
        arr->op = std::make_shared<O>(shared_from_this(), broadcasted_rhs, true);
        return arr;
    }

    template <class O>
    std::shared_ptr<Array> Array::cmp(std::shared_ptr<Array> rhs)
    {
        auto dummy_op = std::make_shared<O>(nullptr, nullptr);
        auto &rview = rhs->get_view();
        if (!shape.broadcastable(rview))
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
        auto broadcasted_lhs = broadcast(rview);
        auto broadcasted_rhs = rhs->broadcast(get_view());
        auto arr = std::make_shared<Array>(Shape(broadcasted_lhs->get_view()), b8, device);
        arr->op = std::make_shared<O>(broadcasted_lhs, broadcasted_rhs);
        return arr;
    }

    template <class O>
    std::shared_ptr<Array> Array::unary_ss(bool in_place)
    {
        auto dummy_op = std::make_shared<O>(nullptr, in_place);
        if (in_place && constant)
        {
            throw CannotUpdateConstArray(id.str());
        }
        if (!unary_dtypes.contains(dtype))
        {
            throw IncompatDtypeForOp(dummy_op->get_name_str(), dtype.str());
        }
        auto arr = std::make_shared<Array>(Shape(get_view()), dtype, device);
        arr->op = std::make_shared<O>(shared_from_this(), in_place);
        return arr;
    }

    template <class O>
    std::shared_ptr<Array> Array::unary_ss_float(bool in_place)
    {
        auto dummy_op = std::make_shared<O>(nullptr, in_place);
        if (in_place)
        {
            if (constant)
            {
                throw CannotUpdateConstArray(id.str());
            }
            else if (dtype != f32)
            {
                throw std::runtime_error("Array " + id.str() + " must be a float array for " + dummy_op->get_name_str() + " operation.");
            }
        }
        auto result_dtype = unary_float_dtypes.find(dtype);
        if (result_dtype == unary_float_dtypes.end())
        {
            throw IncompatDtypeForOp(dummy_op->get_name_str(), dtype.str());
        }
        auto arr = std::make_shared<Array>(Shape(get_view()), result_dtype->second, device);
        arr->op = std::make_shared<O>(shared_from_this(), in_place);
        return arr;
    }

    std::shared_ptr<Array> Array::from_buff(uint8_t *ptr, uint64_t nbytes, const Shape &shape, const Dtype &dtype, const Device &device, bool constant)
    {
        auto op = std::make_shared<BuffOp>();
        auto arr = std::make_shared<Array>(ptr, nbytes, shape, dtype, device, constant);
        arr->op = op;
        return arr;
    }

    std::shared_ptr<Array> Array::from_numpy(uint8_t *ptr, uint64_t nbytes, const Shape &shape, const Dtype &dtype, const Device &device, bool constant)
    {
        auto op = std::make_shared<NumpyOp>();
        auto arr = std::make_shared<Array>(ptr, nbytes, shape, dtype, device, constant);
        arr->op = op;
        return arr;
    }

    std::shared_ptr<Array> Array::add(std::shared_ptr<Array> rhs) { return binary_ss<AddOp>(rhs); }

    std::shared_ptr<Array> Array::self_add(std::shared_ptr<Array> rhs) { return self_binary_ss<AddOp>(rhs); }

    std::shared_ptr<Array> Array::sub(std::shared_ptr<Array> rhs) { return binary_ss<SubOp>(rhs); }

    std::shared_ptr<Array> Array::self_sub(std::shared_ptr<Array> rhs) { return self_binary_ss<SubOp>(rhs); }

    std::shared_ptr<Array> Array::mul(std::shared_ptr<Array> rhs) { return binary_ss<MulOp>(rhs); }

    std::shared_ptr<Array> Array::self_mul(std::shared_ptr<Array> rhs) { return self_binary_ss<MulOp>(rhs); }

    std::shared_ptr<Array> Array::div(std::shared_ptr<Array> rhs) { return binary_ss<DivOp>(rhs); }

    std::shared_ptr<Array> Array::self_div(std::shared_ptr<Array> rhs) { return self_binary_ss<DivOp>(rhs); }

    std::shared_ptr<Array> Array::matmul(std::shared_ptr<Array> rhs)
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
            std::multiplies<uint64_t>());
        std::vector<uint64_t> mm_lview = {
            batch,
            broadcasted_lview[broadcasted_lview.size() - 2],
            broadcasted_lview[broadcasted_lview.size() - 1]};
        std::vector<uint64_t> mm_rview = {
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

    std::shared_ptr<Array> Array::eq(std::shared_ptr<Array> rhs) { return cmp<EqOp>(rhs); }

    std::shared_ptr<Array> Array::neq(std::shared_ptr<Array> rhs) { return cmp<NeqOp>(rhs); }

    std::shared_ptr<Array> Array::lt(std::shared_ptr<Array> rhs) { return cmp<LtOp>(rhs); }

    std::shared_ptr<Array> Array::gt(std::shared_ptr<Array> rhs) { return cmp<GtOp>(rhs); }

    std::shared_ptr<Array> Array::leq(std::shared_ptr<Array> rhs) { return cmp<LeqOp>(rhs); }

    std::shared_ptr<Array> Array::geq(std::shared_ptr<Array> rhs) { return cmp<GeqOp>(rhs); }

    std::shared_ptr<Array> Array::sq(bool in_place) { return unary_ss<SqOp>(in_place); }

    std::shared_ptr<Array> Array::sqrt(bool in_place) { return unary_ss_float<SqrtOp>(in_place); }

    std::shared_ptr<Array> Array::exp(bool in_place) { return unary_ss<ExpOp>(in_place); }

    std::shared_ptr<Array> Array::log(bool in_place) { return unary_ss_float<LogOp>(in_place); }

    std::shared_ptr<Array> Array::neg(bool in_place) { return unary_ss<NegOp>(in_place); }

    std::shared_ptr<Array> Array::recip(bool in_place) { return unary_ss_float<RecipOp>(in_place); }

    std::shared_ptr<Array> Array::reshape(const std::vector<uint64_t> &view)
    {
        if (get_view() == view)
        {
            return shared_from_this();
        }
        Shape::check_view(view);
        uint64_t numel = std::accumulate(view.begin(), view.end(), 1, std::multiplies<uint64_t>());
        if (shape.get_numel() != numel)
        {
            throw std::invalid_argument("Cannot reshape array of " + std::to_string(shape.get_numel()) +
                                        " to " + std::to_string(numel) + " elements.");
        }
        auto arr = std::make_shared<Array>(shape.reshape(view), dtype, device);
        arr->op = std::make_shared<ReshapeOp>(shared_from_this(), view);
        return arr;
    }

    std::shared_ptr<Array> Array::broadcast(const std::vector<uint64_t> &view)
    {
        if (get_view() == view)
        {
            return shared_from_this();
        }
        auto arr = std::make_shared<Array>(shape.broadcast(view), dtype, device);
        arr->op = std::make_shared<BroadcastOp>(shared_from_this(), view);
        return arr;
    }

    std::shared_ptr<Array> Array::broadcast_to(const std::vector<uint64_t> &view)
    {
        if (get_view() == view)
        {
            return shared_from_this();
        }
        auto arr = std::make_shared<Array>(shape.broadcast_to(view), dtype, device);
        arr->op = std::make_shared<BroadcastOp>(shared_from_this(), view);
        return arr;
    }

    std::shared_ptr<Array> Array::copy()
    {
        auto arr = std::make_shared<Array>(Shape(get_view()), dtype, device);
        arr->op = std::make_shared<CopyOp>(shared_from_this());
        return arr;
    }

    std::shared_ptr<Array> Array::permute(const std::vector<uint64_t> &order)
    {
        auto arr = std::make_shared<Array>(shape.permute(order), dtype, device);
        arr->op = std::make_shared<PermuteOp>(shared_from_this(), order);
        return arr;
    }

    std::shared_ptr<Array> Array::T(uint64_t start_dim, uint64_t end_dim)
    {
        check_dims(start_dim, end_dim);
        std::vector<uint64_t> order(get_ndim());
        std::iota(order.begin(), order.begin() + start_dim, 0);
        std::iota(order.begin() + end_dim + 1, order.end(), end_dim + 1);
        // Fill with decreasing values
        std::generate(order.begin() + start_dim, order.begin() + end_dim + 1, [n = end_dim]() mutable
                      { return n--; });
        return permute(order);
    }

    std::shared_ptr<Array> Array::flatten(uint64_t start_dim, uint64_t end_dim)
    {
        check_dims(start_dim, end_dim);
        auto view = get_view();
        auto prod = std::accumulate(view.begin() + start_dim, view.begin() + end_dim + 1, 1ULL, std::multiplies<uint64_t>());
        // Erase from start_dim + 1 to end_dim + 1
        view.erase(view.begin() + start_dim + 1, view.begin() + end_dim + 1);
        // Update view at start_dim
        view[start_dim] = prod;
        return reshape(view);
    }

    std::shared_ptr<Array> Array::sum()
    {
        // Reduce to one element for now
        auto arr = std::make_shared<Array>(Shape({1}), dtype, device);
        arr->op = std::make_shared<SumOp>(shared_from_this());
        return arr;
    }
}
