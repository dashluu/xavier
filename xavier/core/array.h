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

        template <class O>
        std::shared_ptr<Array> unary_ss(bool in_place);

        template <class O>
        std::shared_ptr<Array> unary_ss_float(bool in_place);

        template <class O>
        std::shared_ptr<Array> binary_ss(std::shared_ptr<Array> rhs);

        template <class O>
        std::shared_ptr<Array> self_binary_ss(std::shared_ptr<Array> rhs);

        template <class O>
        std::shared_ptr<Array> cmp(std::shared_ptr<Array> rhs);

        void check_dims(uint64_t start_dim, uint64_t end_dim) const;

    public:
        std::shared_ptr<Array> grad = nullptr;

        Array(uint8_t *ptr, uint64_t nbytes, const Shape &shape, const Dtype &dtype = f32, const Device &device = device0, bool constant = false) : id(id_gen.generate()), shape(shape), dtype(dtype), device(device), constant(constant)
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

        void init_grad()
        {
            // This method only initializes the gradient array without allocating any new buffer for the data
            if (grad == nullptr)
            {
                if (!float_dtypes.contains(dtype))
                {
                    throw std::runtime_error("Only arrays of floating-point types can have gradients but array " + id.str() + " has type " + dtype.str());
                }
                grad = Array::zeros(get_view(), unary_float_dtypes.at(dtype), device);
            }
        }

        void update_grad(std::shared_ptr<Array> grad, bool sub = false) { this->grad = sub ? this->grad->self_sub(grad) : this->grad->self_add(grad); }

        Array(const Array &arr) : id(id_gen.generate()), shape(arr.shape), dtype(arr.dtype), device(arr.device), buff(arr.buff)
        {
        }

        const Id &get_id() const { return id; }

        const Shape &get_shape() const { return shape; }

        uint64_t get_offset() const { return shape.get_offset(); }

        const std::vector<uint64_t> &get_view() const { return shape.get_view(); }

        const std::vector<int64_t> &get_stride() const { return shape.get_stride(); }

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

        uint64_t get_numel() const { return shape.get_numel(); }

        uint64_t get_ndim() const { return shape.get_ndim(); }

        uint64_t get_itemsize() const { return dtype.get_size(); }

        uint64_t get_nbytes() const { return get_numel() * get_itemsize(); }

        // Get the number of bytes of the buffer the array is working on
        // Note: get_buff_nbytes() != get_nbytes()
        uint64_t get_buff_nbytes() const { return buff->get_nbytes(); }

        bool is_contiguous() const { return shape.is_contiguous(); }

        /**
         * @brief Gets a pointer to the k-th element using the array's stride
         * @param k The index of the element to access
         * @return Pointer to the k-th element in the strided array
         */
        uint8_t *strided_idx(uint64_t k) const;

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
        static std::shared_ptr<Array> full_like(std::shared_ptr<Array> arr, int c, const Device &device = device0, bool constant = false)
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
        static std::shared_ptr<Array> full_like(std::shared_ptr<Array> arr, float c, const Device &device = device0, bool constant = false)
        {
            return full(arr->get_view(), c, arr->get_dtype(), device, constant);
        }

        static std::shared_ptr<Array> zeros_like(std::shared_ptr<Array> arr, const Device &device = device0, bool constant = false)
        {
            return full_like(arr, 0, device, constant);
        }

        static std::shared_ptr<Array> ones_like(std::shared_ptr<Array> arr, const Device &device = device0, bool constant = false)
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
        std::shared_ptr<Array> slice(const std::vector<Range> &ranges);

        // Unslice operation yields constant array(for efficiency)
        std::shared_ptr<Array> unslice(const Shape &orig_shape, const std::vector<Range> &ranges);

        static std::shared_ptr<Array> arange(const std::vector<uint64_t> &view, int64_t start, int64_t step, const Dtype &dtype = f32, const Device &device = device0, bool constant = false);

        static std::shared_ptr<Array> full(const std::vector<uint64_t> &view, int c, const Dtype &dtype = f32, const Device &device = device0, bool constant = false);

        static std::shared_ptr<Array> full(const std::vector<uint64_t> &view, float c, const Dtype &dtype = f32, const Device &device = device0, bool constant = false);

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
        static std::shared_ptr<Array> full(int c, const Dtype &dtype = f32, const Device &device = device0)
        {
            std::vector<uint64_t> view = {1};
            return full(view, c, dtype, device, true);
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
        static std::shared_ptr<Array> full(float c, const Dtype &dtype = f32, const Device &device = device0)
        {
            std::vector<uint64_t> view = {1};
            return full(view, c, dtype, device, true);
        }

        static std::shared_ptr<Array> zeros(const std::vector<uint64_t> &view, const Dtype &dtype = f32, const Device &device = device0, bool constant = false)
        {
            return full(view, 0, dtype, device, constant);
        }

        static std::shared_ptr<Array> ones(const std::vector<uint64_t> &view, const Dtype &dtype = f32, const Device &device = device0, bool constant = false)
        {
            return full(view, 1, dtype, device, constant);
        }

        static std::shared_ptr<Array> from_buff(uint8_t *ptr, uint64_t nbytes, const Shape &shape, const Dtype &dtype = f32, const Device &device = device0, bool constant = false);

        static std::shared_ptr<Array> from_numpy(uint8_t *ptr, uint64_t nbytes, const Shape &shape, const Dtype &dtype = f32, const Device &device = device0, bool constant = false);

        std::shared_ptr<Array> add(std::shared_ptr<Array> rhs);

        std::shared_ptr<Array> self_add(std::shared_ptr<Array> rhs);

        std::shared_ptr<Array> sub(std::shared_ptr<Array> rhs);

        std::shared_ptr<Array> self_sub(std::shared_ptr<Array> rhs);

        std::shared_ptr<Array> mul(std::shared_ptr<Array> rhs);

        std::shared_ptr<Array> mul(int c) { return mul(full(c, dtype, device)); }

        std::shared_ptr<Array> mul(float c) { return mul(full(c, dtype, device)); }

        std::shared_ptr<Array> self_mul(std::shared_ptr<Array> rhs);

        std::shared_ptr<Array> div(std::shared_ptr<Array> rhs);

        std::shared_ptr<Array> self_div(std::shared_ptr<Array> rhs);

        std::shared_ptr<Array> matmul(std::shared_ptr<Array> rhs);

        std::shared_ptr<Array> eq(std::shared_ptr<Array> rhs);

        std::shared_ptr<Array> neq(std::shared_ptr<Array> rhs);

        std::shared_ptr<Array> lt(std::shared_ptr<Array> rhs);

        std::shared_ptr<Array> gt(std::shared_ptr<Array> rhs);

        std::shared_ptr<Array> leq(std::shared_ptr<Array> rhs);

        std::shared_ptr<Array> geq(std::shared_ptr<Array> rhs);

        std::shared_ptr<Array> sq(bool in_place = false);

        std::shared_ptr<Array> sqrt(bool in_place = false);

        std::shared_ptr<Array> exp(bool in_place = false);

        std::shared_ptr<Array> log(bool in_place = false);

        std::shared_ptr<Array> neg(bool in_place = false);

        std::shared_ptr<Array> recip(bool in_place = false);

        std::shared_ptr<Array> reshape(const std::vector<uint64_t> &view);

        std::shared_ptr<Array> broadcast(const std::vector<uint64_t> &view);

        std::shared_ptr<Array> broadcast_to(const std::vector<uint64_t> &view);

        std::shared_ptr<Array> copy();

        std::shared_ptr<Array> permute(const std::vector<uint64_t> &order);

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
        std::shared_ptr<Array> T(uint64_t start_dim, uint64_t end_dim);

        /**
         * @brief Transposes array starting from a specified dimension to the last dimension.
         *
         * @param start_dim Starting dimension for transpose operation (0-based index)
         * @return std::shared_ptr<Array> A new array with dimensions transposed from start_dim to the last dimension
         * @note This is a convenience overload that calls T(start_dim, get_ndim() - 1)
         */
        std::shared_ptr<Array> T(uint64_t start_dim)
        {
            return T(start_dim, get_ndim() - 1);
        }

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
        std::shared_ptr<Array> flatten(uint64_t start_dim, uint64_t end_dim);

        std::shared_ptr<Array> as_contiguous() { return is_contiguous() ? shared_from_this() : copy(); }

        std::shared_ptr<Array> sum();
    };

    inline IdGenerator Array::id_gen = IdGenerator();
}