#pragma once

#include "exceptions.h"
#include "range.h"
#include "shape.h"
#include "dtype.h"
#include "device.h"
#include "ops.h"
#include "buffer.h"
#include "id.h"

namespace xv::core
{
    class Array : public std::enable_shared_from_this<Array>, public IStr
    {
    private:
        static IdGenerator id_gen;
        Id id;
        Shape shape;
        Dtype dtype = f32;
        Device device;
        std::shared_ptr<Buffer> buff = nullptr;
        std::shared_ptr<Op> op = nullptr;
        bool constant;

        template <class I, class F>
        std::string fmt_num(uint8_t *ptr, Dtype dtype) const
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

        std::string fmt(uint8_t *ptr, const Dtype &dtype) const;

        template <class O>
        ArrayPtr unary_ss(bool in_place)
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
        ArrayPtr unary_ss_float(bool in_place)
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

        template <class O>
        ArrayPtr binary_ss(ArrayPtr rhs)
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
        ArrayPtr self_binary_ss(ArrayPtr rhs)
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
        ArrayPtr cmp(ArrayPtr rhs)
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
        ArrayPtr reduce(const std::vector<usize> &dims)
        {
            ArrayPtr arr;
            if (dims.size() == 0)
            {
                // Reduce to one element
                arr = std::make_shared<Array>(Shape({1}), dtype, device);
            }
            else
            {
                // Assume 2D matrix for now
                Shape reduced_shape({get_view()[0], 1});
                arr = std::make_shared<Array>(reduced_shape, dtype, device);
            }
            arr->op = std::make_shared<O>(shared_from_this(), dims);
            return arr;
        }

        template <class T>
        void check_scalar(T c) { static_assert(std::is_arithmetic_v<T>, "Scalar type must be numeric"); }

        void check_dims(usize start_dim, usize end_dim) const;

    public:
        ArrayPtr grad = nullptr;
        ArrayPtr grad_root = nullptr;

        Array(uint8_t *ptr, usize nbytes, const Shape &shape, const Dtype &dtype = f32, const Device &device = device0, bool constant = false) : id(id_gen.generate()), shape(shape), dtype(dtype), device(device), constant(constant)
        {
            buff = std::make_shared<Buffer>(ptr, nbytes, false);
        }

        Array(const Shape &shape, const Dtype &dtype = f32, const Device &device = device0, bool constant = false) : id(id_gen.generate()), shape(shape), dtype(dtype), device(device), constant(constant)
        {
        }

        ~Array()
        {
            if (buff != nullptr)
            {
                device.get_allocator()->free(buff);
            }
        }

        // Only allocate if the buffer is null
        void alloc()
        {
            if (buff == nullptr)
            {
                buff = device.get_allocator()->alloc(get_nbytes());
            }
        }

        // Low overhead by pointing to another buffer
        void alloc(Buffer &buff)
        {
            if (this->buff == nullptr)
            {
                this->buff = std::make_shared<Buffer>(buff);
            }
        }

        void init_grad(bool is_root = false)
        {
            // This method only initializes the gradient array without allocating any new buffer for the data
            if (grad == nullptr)
            {
                if (!float_dtypes.contains(dtype))
                {
                    throw std::runtime_error("Only arrays of floating-point types can have gradients but array " + id.str() + " has type " + dtype.str());
                }
                const Dtype &grad_dtype = unary_float_dtypes.at(dtype);
                grad = is_root ? ones(get_view(), grad_dtype, device) : zeros(get_view(), grad_dtype, device);
            }
        }

        void update_grad(ArrayPtr grad, bool sub = false)
        {
            this->grad = sub ? this->grad->self_sub(grad) : this->grad->self_add(grad);
            this->grad_root = this->grad;
        }

        Array(const Array &arr) : id(id_gen.generate()), shape(arr.shape), dtype(arr.dtype), device(arr.device), buff(arr.buff)
        {
        }

        const Id &get_id() const { return id; }

        const Shape &get_shape() const { return shape; }

        usize get_offset() const { return shape.get_offset(); }

        const ShapeView &get_view() const { return shape.get_view(); }

        const ShapeStride get_stride() const { return shape.get_stride(); }

        // Gets the buffer pointer without accounting for offset
        uint8_t *get_buff_ptr() const { return buff->get_ptr(); }

        // Gets the buffer pointer after accounting for offset
        uint8_t *get_ptr() const { return get_buff_ptr() + get_offset() * get_itemsize(); }

        const Dtype &get_dtype() const { return dtype; }

        const Device &get_device() const { return device; }

        bool is_constant() const { return constant; }

        void set_constant() { constant = true; }

        std::shared_ptr<Buffer> get_buff() const { return buff; }

        std::shared_ptr<Op> get_op() const { return op; }

        usize get_numel() const { return shape.get_numel(); }

        usize get_ndim() const { return shape.get_ndim(); }

        usize get_itemsize() const { return dtype.get_size(); }

        usize get_nbytes() const { return get_numel() * get_itemsize(); }

        // Get the number of bytes of the buffer the array is working on
        // Note: get_buff_nbytes() != get_nbytes()
        usize get_buff_nbytes() const { return buff->get_nbytes(); }

        bool is_contiguous() const { return shape.is_contiguous(); }

        /**
         * @brief Gets a pointer to the k-th element using the array's stride
         * @param k The index of the element to access
         * @return Pointer to the k-th element in the strided array
         */
        uint8_t *strided_idx(usize k) const;

        // TODO: implement this
        Array &operator=(const Array &arr) = delete;

        const std::string str() const override;

        /**
         * @brief Creates a new array with the same shape as the input array, filled with an integer value.
         *
         * @param arr Input array whose shape will be used as reference
         * @param c Constant value to fill the new array with
         * @param device Device where the new array will be allocated (default: device0)
         * @param constant Whether the array should be marked as constant (default: false)
         * @return std::shared_ptr<Array> New array with same shape as input, filled with value c
         */
        static ArrayPtr full_like(ArrayPtr arr, int c, const Device &device = device0, bool constant = false)
        {
            return full(arr->get_view(), c, arr->get_dtype(), device, constant);
        }

        /**
         * @brief Creates a new array with the same shape as input array, filled with a floating-point value.
         *
         * @param arr Input array whose shape will be used as reference
         * @param c Scalar value to fill the new array with
         * @param device Target device where the array will be allocated (defaults to device0)
         * @param constant Whether the array should be marked as constant (defaults to false)
         *
         * @return std::shared_ptr<Array> New array with same shape as input, filled with value c
         *
         * @see full()
         */
        static ArrayPtr full_like(ArrayPtr arr, float c, const Device &device = device0, bool constant = false)
        {
            return full(arr->get_view(), c, arr->get_dtype(), device, constant);
        }

        static ArrayPtr zeros_like(ArrayPtr arr, const Device &device = device0, bool constant = false)
        {
            return full_like(arr, 0, device, constant);
        }

        static ArrayPtr ones_like(ArrayPtr arr, const Device &device = device0, bool constant = false)
        {
            return full_like(arr, 1, device, constant);
        }

        /**
         * Creates a new Array containing elements selected by the given ranges.
         *
         * @param ranges A vector of Range objects specifying the indices to select from each dimension.
         *              Each Range object defines the start, stop, and step values for slicing along that dimension.
         *
         * @return A shared pointer to a new Array containing the selected elements.
         *         The returned Array shares the same underlying buffer as the original Array,
         *         but with modified shape and strides to represent the slice view,
         *         avoiding memory allocation for efficiency.
         *
         * @throws std::invalid_argument If the number of ranges doesn't match the Array's dimensions
         *                              or if any range specifies invalid indices.
         */
        ArrayPtr slice(const std::vector<Range> &ranges);

        static ArrayPtr arange(const ShapeView &view, isize start, isize step, const Dtype &dtype = f32, const Device &device = device0, bool constant = false);

        static ArrayPtr full(const ShapeView &view, int c, const Dtype &dtype = f32, const Device &device = device0, bool constant = false);

        static ArrayPtr full(const ShapeView &view, float c, const Dtype &dtype = f32, const Device &device = device0, bool constant = false);

        // Saves memory by storing only one constant value and broadcasts later if needed
        /**
         * @brief Creates a scalar array filled with a specified integer value
         *
         * @param c The integer value to fill the array with
         * @param dtype The data type of the array elements (default: float32)
         * @param device The device where the array will be allocated (default: device0)
         *
         * @return std::shared_ptr<Array> A shared pointer to the newly created array
         *
         * @details This is a convenience function that creates a scalar (1-dimensional array of size 1)
         * filled with the specified value. It internally calls the more general full() function
         * with a shape vector of {1}.
         */
        static ArrayPtr full(int c, const Dtype &dtype = f32, const Device &device = device0)
        {
            return full({1}, c, dtype, device, true);
        }

        /**
         * @brief Creates a scalar array filled with a specified floating-point value
         *
         * @param c The floating-point value to fill the array with
         * @param dtype The data type of the array elements (default: float32)
         * @param device The device where the array will be allocated (default: device0)
         *
         * @return std::shared_ptr<Array> A shared pointer to the newly created array
         *
         * @details This is a convenience function that creates a scalar (1-dimensional array of size 1)
         * filled with the specified value. It internally calls the more general full() function
         * with a shape vector of {1}.
         */
        static ArrayPtr full(float c, const Dtype &dtype = f32, const Device &device = device0)
        {
            return full({1}, c, dtype, device, true);
        }

        static ArrayPtr zeros(const ShapeView &view, const Dtype &dtype = f32, const Device &device = device0, bool constant = false)
        {
            return full(view, 0, dtype, device, constant);
        }

        static ArrayPtr ones(const ShapeView &view, const Dtype &dtype = f32, const Device &device = device0, bool constant = false)
        {
            return full(view, 1, dtype, device, constant);
        }

        static ArrayPtr from_buff(uint8_t *ptr, usize nbytes, const Shape &shape, const Dtype &dtype = f32, const Device &device = device0, bool constant = false);

        static ArrayPtr from_numpy(uint8_t *ptr, usize nbytes, const Shape &shape, const Dtype &dtype = f32, const Device &device = device0, bool constant = false);

        ArrayPtr add(ArrayPtr rhs) { return binary_ss<AddOp>(rhs); }

        template <typename T>
        ArrayPtr add(T c)
        {
            check_scalar(c);
            return add(full(c, dtype, device));
        }

        ArrayPtr self_add(ArrayPtr rhs) { return self_binary_ss<AddOp>(rhs); }

        template <typename T>
        ArrayPtr self_add(T c)
        {
            check_scalar(c);
            return self_add(full(c, dtype, device));
        }

        ArrayPtr sub(ArrayPtr rhs) { return binary_ss<SubOp>(rhs); }

        template <typename T>
        ArrayPtr sub(T c)
        {
            check_scalar(c);
            return sub(full(c, dtype, device));
        }

        ArrayPtr self_sub(ArrayPtr rhs) { return self_binary_ss<SubOp>(rhs); }

        template <typename T>
        ArrayPtr self_sub(T c)
        {
            check_scalar(c);
            return self_sub(full(c, dtype, device));
        }

        ArrayPtr mul(ArrayPtr rhs) { return binary_ss<MulOp>(rhs); }

        template <typename T>
        ArrayPtr mul(T c)
        {
            check_scalar(c);
            return mul(full(c, dtype, device));
        }

        ArrayPtr self_mul(ArrayPtr rhs) { return self_binary_ss<MulOp>(rhs); }

        template <typename T>
        ArrayPtr self_mul(T c)
        {
            check_scalar(c);
            return self_mul(full(c, dtype, device));
        }

        ArrayPtr div(ArrayPtr rhs) { return binary_ss<DivOp>(rhs); }

        template <typename T>
        ArrayPtr div(T c)
        {
            check_scalar(c);
            return div(full(c, dtype, device));
        }

        ArrayPtr self_div(ArrayPtr rhs) { return self_binary_ss<DivOp>(rhs); }

        template <typename T>
        ArrayPtr self_div(T c)
        {
            check_scalar(c);
            return self_div(full(c, dtype, device));
        }

        ArrayPtr matmul(ArrayPtr rhs);

        template <typename T>
        ArrayPtr matmul(T c)
        {
            check_scalar(c);
            return matmul(full(c, dtype, device));
        }

        ArrayPtr eq(ArrayPtr rhs) { return cmp<EqOp>(rhs); }

        template <typename T>
        ArrayPtr eq(T c)
        {
            check_scalar(c);
            return eq(full(c, dtype, device));
        }

        ArrayPtr neq(ArrayPtr rhs) { return cmp<NeqOp>(rhs); }

        template <typename T>
        ArrayPtr neq(T c)
        {
            check_scalar(c);
            return neq(full(c, dtype, device));
        }

        ArrayPtr lt(ArrayPtr rhs) { return cmp<LtOp>(rhs); }

        template <typename T>
        ArrayPtr lt(T c)
        {
            check_scalar(c);
            return lt(full(c, dtype, device));
        }

        ArrayPtr gt(ArrayPtr rhs) { return cmp<GtOp>(rhs); }

        template <typename T>
        ArrayPtr gt(T c)
        {
            check_scalar(c);
            return gt(full(c, dtype, device));
        }

        ArrayPtr leq(ArrayPtr rhs) { return cmp<LeqOp>(rhs); }

        template <typename T>
        ArrayPtr leq(T c)
        {
            check_scalar(c);
            return leq(full(c, dtype, device));
        }

        ArrayPtr geq(ArrayPtr rhs) { return cmp<GeqOp>(rhs); }

        template <typename T>
        ArrayPtr geq(T c)
        {
            check_scalar(c);
            return geq(full(c, dtype, device));
        }

        ArrayPtr sq(bool in_place = false) { return unary_ss<SqOp>(in_place); }

        ArrayPtr sqrt(bool in_place = false) { return unary_ss_float<SqrtOp>(in_place); }

        ArrayPtr exp(bool in_place = false) { return unary_ss<ExpOp>(in_place); }

        ArrayPtr log(bool in_place = false) { return unary_ss_float<LogOp>(in_place); }

        ArrayPtr neg(bool in_place = false) { return unary_ss<NegOp>(in_place); }

        ArrayPtr recip(bool in_place = false) { return unary_ss_float<RecipOp>(in_place); }

        ArrayPtr reshape(const ShapeView &view);

        bool copy_when_reshape() { return !is_contiguous(); }

        ArrayPtr broadcast(const ShapeView &view);

        ArrayPtr broadcast_to(const ShapeView &view);

        ArrayPtr identity();

        ArrayPtr permute(const ShapeOrder &order);

        /**
         * @brief Transposes the dimensions of the array between `start_dim` and `end_dim`.
         *
         * This function rearranges the dimensions of the array such that the dimensions
         * between `start_dim` and `end_dim` (inclusive) are reversed, while the other
         * dimensions remain in their original order. It uses the `permute` function to
         * achieve the reordering.
         *
         * @param start_dim The starting dimension index for the transpose operation.
         * @param end_dim The ending dimension index for the transpose operation.
         * @return A new Array object with the transposed dimensions.
         * @throws std::invalid_argument if `start_dim` or `end_dim` are out of bounds or
         *         if `start_dim` is greater than `end_dim`.
         */
        ArrayPtr T(usize start_dim, usize end_dim);

        /**
         * @brief Transposes array starting from a specified dimension to the last dimension.
         *
         * @param start_dim Starting dimension for transpose operation (0-based index)
         * @return std::shared_ptr<Array> A new array with dimensions transposed from start_dim to the last dimension
         * @note This is a convenience overload that calls T(start_dim, get_ndim() - 1)
         */
        ArrayPtr T(usize start_dim) { return T(start_dim, get_ndim() - 1); }

        /**
         * @brief Flattens a multi-dimensional array by collapsing dimensions from start_dim to end_dim into a single dimension.
         *
         * @param start_dim Starting dimension to flatten (inclusive)
         * @param end_dim Ending dimension to flatten (inclusive)
         * @return std::shared_ptr<Array> A new array with the specified dimensions flattened into one
         *
         * @details This method preserves the order of elements while reducing the number of dimensions.
         * The dimensions from start_dim to end_dim (inclusive) are merged into a single dimension,
         * while dimensions before start_dim and after end_dim remain unchanged.
         *
         * @throws std::invalid_argument If start_dim > end_dim or if end_dim exceeds array dimensions
         */
        ArrayPtr flatten(usize start_dim, usize end_dim);

        ArrayPtr as_contiguous() { return is_contiguous() ? shared_from_this() : identity(); }

        ArrayPtr sum(const std::vector<usize> &dims = {}) { return reduce<SumOp>(dims); }

        ArrayPtr max(const std::vector<usize> &dims = {}) { return reduce<MaxOp>(dims); }

        ArrayPtr min(const std::vector<usize> &dims = {}) { return reduce<MinOp>(dims); }
    };

    inline IdGenerator Array::id_gen = IdGenerator();
}