#pragma once

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
        std::shared_ptr<Array> grad = nullptr;

        void check_ranges(const std::vector<Range> &ranges) const;

    public:
        Array(uint8_t *ptr, const Shape &shape, const Dtype &dtype = f32, const Device &device = device0) : shape(shape), dtype(dtype), device(device)
        {
            // No information on the number of bytes allocated so 0 is used
            buff = std::make_shared<Buffer>(ptr + shape.get_offset() * dtype.get_size(), 0, false);
            id = id_gen.generate();
        }

        Array(const Shape &shape, const Dtype &dtype = f32, const Device &device = device0) : shape(shape), dtype(dtype), device(device)
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

        void alloc()
        {
            if (buff == nullptr)
            {
                buff = device.get_allocator()->alloc(get_nbytes());
            }
        }

        Array(const Array &arr) : shape(arr.shape), dtype(arr.dtype), device(arr.device), buff(arr.buff) {}

        const IdType &get_id() const { return id; }

        const Shape &get_shape() const { return shape; }

        uint8_t *get_ptr() const { return buff->get_ptr(); }

        const Dtype &get_dtype() const { return dtype; }

        const Device &get_device() const { return device; }

        std::shared_ptr<Op> get_op() const { return op; }

        std::shared_ptr<Array> get_grad() const { return grad; }

        uint64_t get_numel() const { return shape.get_numel(); }

        uint64_t get_nbytes() const { return shape.get_numel() * dtype.get_size(); }

        bool is_contiguous() const { return shape.is_contiguous(); }

        // TODO: implement this
        Array &operator=(const Array &arr) = delete;

        const std::string str() const override;

        void copy_to(std::shared_ptr<const Array> dest) const;

        std::shared_ptr<Array> interpret_(const Dtype &dtype);

        std::shared_ptr<Array> slice(const std::vector<Range> &ranges);

        static std::shared_ptr<Array> arange(const std::vector<uint64_t> &view, int64_t start, int64_t step, const Dtype &dtype = f32, const Device &device = device0)
        {
            auto op = std::make_shared<ArangeOp>(view, start, step, dtype);
            auto arr = std::make_shared<Array>(Shape(view), dtype, device);
            arr->op = op;
            return arr;
        }

        static std::shared_ptr<Array> constant(const std::vector<uint64_t> &view, double c, const Dtype &dtype = f32, const Device &device = device0)
        {
            auto op = std::make_shared<ConstOp>(view, c, dtype);
            auto arr = std::make_shared<Array>(Shape(view), dtype, device);
            arr->op = op;
            return arr;
        }

        template <class O>
        std::shared_ptr<Array> binary(std::shared_ptr<Array> rhs, const std::unordered_map<Dtype, Dtype> &dtype_map);

        std::shared_ptr<Array> add(std::shared_ptr<Array> rhs);

        std::shared_ptr<Array> sub(std::shared_ptr<Array> rhs);

        std::shared_ptr<Array> mul(std::shared_ptr<Array> rhs);

        std::shared_ptr<Array> div(std::shared_ptr<Array> rhs);

        std::shared_ptr<Array> reshape_(const std::vector<uint64_t> &view);
    };

    inline IdGenerator Array::id_gen = IdGenerator();
}