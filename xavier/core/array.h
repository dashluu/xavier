#pragma once

#include "exceptions.h"
#include "range.h"
#include "shape.h"
#include "dtype.h"
#include "device.h"
#include "ops.h"
#include "buffer.h"

namespace xv::core
{
    using IdType = uint64_t;

    struct IdGenerator
    {
    private:
        static IdType counter;

    public:
        IdGenerator() = default;
        IdGenerator(const IdGenerator &) = delete;
        IdGenerator &operator=(const IdGenerator &) = delete;

        IdType generate()
        {
            IdType curr = counter;
            counter++;
            return curr;
        }
    };

    inline IdType IdGenerator::counter = 1;

    class Array : public std::enable_shared_from_this<Array>, public IStr
    {
    private:
        static IdGenerator id_gen;
        IdType id;
        Shape shape;
        Dtype dtype = f32;
        Device device;
        std::shared_ptr<Buffer> buff = nullptr;
        std::shared_ptr<Op> op = nullptr;
        bool constant;

        void check_ranges(const std::vector<Range> &ranges) const;

        std::shared_ptr<Array> matmul_broadcast(const std::vector<uint64_t> &view);

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

    public:
        std::shared_ptr<Array> cum_grad = nullptr;
        std::shared_ptr<Array> grad = nullptr;

        Array(uint8_t *ptr, uint64_t nbytes, const Shape &shape, const Dtype &dtype = f32, const Device &device = device0, bool constant = false) : shape(shape), dtype(dtype), device(device), constant(constant)
        {
            buff = std::make_shared<Buffer>(ptr, nbytes, false);
            id = id_gen.generate();
        }

        Array(const Shape &shape, const Dtype &dtype = f32, const Device &device = device0, bool constant = false) : shape(shape), dtype(dtype), device(device), constant(constant)
        {
            id = id_gen.generate();
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
            if (cum_grad == nullptr)
            {
                if (!float_dtypes.contains(dtype))
                {
                    throw std::runtime_error("Only arrays of floating-point types can have gradients but array " + std::to_string(id) + " has type " + dtype.str());
                }
                cum_grad = Array::zeros(shape.get_view(), unary_float_dtypes.at(dtype), device);
            }
        }

        void update_grad() { cum_grad = cum_grad->self_add(grad); }

        Array(const Array &arr) : shape(arr.shape), dtype(arr.dtype), device(arr.device), buff(arr.buff)
        {
            id = id_gen.generate();
        }

        const IdType &get_id() const { return id; }

        const Shape &get_shape() const { return shape; }

        // Gets the buffer pointer without accounting for offset
        uint8_t *get_buff_ptr() const { return buff->get_ptr(); }

        // Gets the buffer pointer after accounting for offset
        uint8_t *get_ptr() const { return get_buff_ptr() + shape.get_offset() * get_itemsize(); }

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

        uint8_t *access_(uint64_t k) const;

        // TODO: implement this
        Array &operator=(const Array &arr) = delete;

        const std::string str() const override;

        static std::shared_ptr<Array> full_like(std::shared_ptr<Array> arr, int c, const Device &device = device0, bool constant = false)
        {
            return full(arr->get_shape().get_view(), c, arr->get_dtype(), device, constant);
        }

        static std::shared_ptr<Array> full_like(std::shared_ptr<Array> arr, float c, const Device &device = device0, bool constant = false)
        {
            return full(arr->get_shape().get_view(), c, arr->get_dtype(), device, constant);
        }

        static std::shared_ptr<Array> zeros_like(std::shared_ptr<Array> arr, const Device &device = device0, bool constant = false)
        {
            return full_like(arr, 0, device, constant);
        }

        static std::shared_ptr<Array> ones_like(std::shared_ptr<Array> arr, const Device &device = device0, bool constant = false)
        {
            return full_like(arr, 1, device, constant);
        }

        std::shared_ptr<Array> slice(const std::vector<Range> &ranges);

        static std::shared_ptr<Array> arange(const std::vector<uint64_t> &view, int64_t start, int64_t step, const Dtype &dtype = f32, const Device &device = device0, bool constant = false);

        static std::shared_ptr<Array> full(const std::vector<uint64_t> &view, int c, const Dtype &dtype = f32, const Device &device = device0, bool constant = false);

        static std::shared_ptr<Array> full(const std::vector<uint64_t> &view, float c, const Dtype &dtype = f32, const Device &device = device0, bool constant = false);

        // Saves memory by storing only one constant value and broadcasts later if needed
        static std::shared_ptr<Array> full(int c, const Dtype &dtype = f32, const Device &device = device0)
        {
            std::vector<uint64_t> view = {1};
            return full(view, c, dtype, device, true);
        }

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

        std::shared_ptr<Array> as_contiguous() { return is_contiguous() ? shared_from_this() : copy(); }
    };

    inline IdGenerator Array::id_gen = IdGenerator();
}