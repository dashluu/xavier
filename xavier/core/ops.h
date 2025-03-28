#pragma once

#include "../common.h"
#include "dtype.h"

namespace xv::core
{
    enum class OpName
    {
        RANDN,
        ARANGE,
        FULL,
        BUFF,
        NUMPY,
        ADD,
        SUB,
        MUL,
        DIV,
        EQ,
        NEQ,
        GT,
        GEQ,
        LT,
        LEQ,
        MATMUL,
        SQ,
        SQRT,
        NEG,
        EXP,
        LOG,
        RECIP,
        RESHAPE,
        PERMUTE,
        BROADCAST,
        SQUEEZE,
        UNSQUEEZE,
        INTERPRET,
        SLICE,
        UNSLICE,
        COPY,
        SUM
    };

    enum class OpType
    {
        INITIALIZER,
        UNARY,
        BINARY,
        TRANSFORM,
        REDUCE,
        MOVE
    };

    inline const std::unordered_map<OpName, std::string> opnames = {
        {OpName::RANDN, "randn"},
        {OpName::ARANGE, "arange"},
        {OpName::FULL, "full"},
        {OpName::BUFF, "buff"},
        {OpName::NUMPY, "numpy"},
        {OpName::ADD, "add"},
        {OpName::SUB, "sub"},
        {OpName::MUL, "mul"},
        {OpName::DIV, "div"},
        {OpName::EQ, "eq"},
        {OpName::NEQ, "neq"},
        {OpName::GT, "gt"},
        {OpName::GEQ, "geq"},
        {OpName::LT, "lt"},
        {OpName::LEQ, "leq"},
        {OpName::MATMUL, "matmul"},
        {OpName::SQ, "sq"},
        {OpName::SQRT, "sqrt"},
        {OpName::NEG, "neg"},
        {OpName::EXP, "exp"},
        {OpName::LOG, "log"},
        {OpName::RECIP, "recip"},
        {OpName::BROADCAST, "broadcast"},
        {OpName::SQUEEZE, "squeeze"},
        {OpName::UNSQUEEZE, "unsqueeze"},
        {OpName::RESHAPE, "reshape"},
        {OpName::PERMUTE, "permute"},
        {OpName::INTERPRET, "interpret"},
        {OpName::SLICE, "slice"},
        {OpName::UNSLICE, "unslice"},
        {OpName::COPY, "copy"},
        {OpName::SUM, "sum"}};

    struct Op : public std::enable_shared_from_this<Op>, public IStr
    {
    protected:
        OpName name;
        OpType type;

        Op(OpName name, OpType type) : name(name), type(type) {}

    public:
        Op(const Op &) = delete;
        Op &operator=(const Op &) = delete;
        virtual ~Op() = default;
        OpName get_name() const { return name; }
        std::string get_name_str() const { return opnames.at(name); }
        OpType get_type() const { return type; }
        virtual void backward(std::shared_ptr<Array> arr) const {}
    };

    struct InitializerOp : public Op
    {
    public:
        InitializerOp(OpName name) : Op(name, OpType::INITIALIZER) {}
    };

    struct ArangeOp : public InitializerOp
    {
    private:
        std::vector<uint64_t> view;
        int64_t start;
        int64_t step;
        Dtype dtype;

    public:
        ArangeOp(const std::vector<uint64_t> &view, int64_t start, int64_t step, const Dtype &dtype) : InitializerOp(OpName::ARANGE), view(view), start(start), step(step), dtype(dtype) {}
        const std::vector<uint64_t> &get_view() { return view; }
        int64_t get_start() { return start; }
        int64_t get_step() { return step; }
        const Dtype &get_dtype() { return dtype; }
        const std::string str() const override
        {
            return get_name_str() + ", view: (" + vnumstr(view) + "), start: " + std::to_string(start) + ", step: " + std::to_string(step);
        }
    };

    struct FullOp : public InitializerOp
    {
    private:
        std::vector<uint64_t> view;
        int c;
        Dtype dtype;

    public:
        FullOp(const std::vector<uint64_t> &view, int c, const Dtype &dtype) : InitializerOp(OpName::FULL), view(view), c(c), dtype(dtype) {}
        const std::vector<uint64_t> &get_view() { return view; }
        int get_const() const { return c; }
        const Dtype &get_dtype() { return dtype; }
        const std::string str() const override
        {
            auto s = get_name_str() + ", view: (" + vnumstr(view) + "), value: ";
            if (bool_dtypes.contains(dtype))
            {
                return s + std::to_string(static_cast<bool>(c));
            }
            else if (int_dtypes.contains(dtype))
            {
                return s + std::to_string(c);
            }
            return s + std::to_string(std::bit_cast<float>(c));
        }
    };

    struct BuffOp : public InitializerOp
    {
    public:
        BuffOp() : InitializerOp(OpName::BUFF) {}
        const std::string str() const override { return get_name_str(); }
    };

    struct NumpyOp : public InitializerOp
    {
    public:
        NumpyOp() : InitializerOp(OpName::NUMPY) {}
        const std::string str() const override { return get_name_str(); }
    };

    struct UnaryOp : public Op
    {
    protected:
        bool in_place;
        std::shared_ptr<Array> operand;

    public:
        UnaryOp(OpName name, std::shared_ptr<Array> operand, bool in_place) : Op(name, OpType::UNARY), operand(operand), in_place(in_place) {}
        std::shared_ptr<Array> get_operand() const { return operand; }
        const std::string str() const override;
        bool is_in_place() const { return in_place; }
    };

    struct BinaryOp : public Op
    {
    protected:
        bool in_place;
        std::shared_ptr<Array> lhs;
        std::shared_ptr<Array> rhs;

    public:
        BinaryOp(OpName name, std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs, bool in_place) : Op(name, OpType::BINARY), lhs(lhs), rhs(rhs), in_place(in_place) {}
        std::shared_ptr<Array> get_lhs() const { return lhs; }
        std::shared_ptr<Array> get_rhs() const { return rhs; }
        const std::string str() const override;
        bool is_in_place() const { return in_place; }
    };

    struct TransformOp : public Op
    {
    protected:
        std::shared_ptr<Array> operand;

    public:
        TransformOp(OpName name, std::shared_ptr<Array> operand) : Op(name, OpType::TRANSFORM), operand(operand) {}
        std::shared_ptr<Array> get_operand() const { return operand; }
        const std::string str() const override;
    };

    struct ReduceOp : public Op
    {
    protected:
        std::shared_ptr<Array> operand;

    public:
        ReduceOp(OpName name, std::shared_ptr<Array> operand) : Op(name, OpType::REDUCE), operand(operand) {}
        std::shared_ptr<Array> get_operand() const { return operand; }
        const std::string str() const override;
    };

    struct AddOp : public BinaryOp
    {
    public:
        AddOp(std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs, bool in_place) : BinaryOp(OpName::ADD, lhs, rhs, in_place) {}

        void backward(std::shared_ptr<Array> arr) const override;
    };

    struct SubOp : public BinaryOp
    {
    public:
        SubOp(std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs, bool in_place) : BinaryOp(OpName::SUB, lhs, rhs, in_place) {}

        void backward(std::shared_ptr<Array> arr) const override;
    };

    struct MulOp : public BinaryOp
    {
    public:
        MulOp(std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs, bool in_place) : BinaryOp(OpName::MUL, lhs, rhs, in_place) {}

        void backward(std::shared_ptr<Array> arr) const override;
    };

    struct DivOp : public BinaryOp
    {
    public:
        DivOp(std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs, bool in_place) : BinaryOp(OpName::DIV, lhs, rhs, in_place) {}

        void backward(std::shared_ptr<Array> arr) const override;
    };

    struct EqOp : public BinaryOp
    {
    public:
        EqOp(std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs) : BinaryOp(OpName::EQ, lhs, rhs, false) {}
    };

    struct NeqOp : public BinaryOp
    {
    public:
        NeqOp(std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs) : BinaryOp(OpName::NEQ, lhs, rhs, false) {}
    };

    struct LtOp : public BinaryOp
    {
    public:
        LtOp(std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs) : BinaryOp(OpName::LT, lhs, rhs, false) {}
    };

    struct GtOp : public BinaryOp
    {
    public:
        GtOp(std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs) : BinaryOp(OpName::GT, lhs, rhs, false) {}
    };

    struct LeqOp : public BinaryOp
    {
    public:
        LeqOp(std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs) : BinaryOp(OpName::LEQ, lhs, rhs, false) {}
    };

    struct GeqOp : public BinaryOp
    {
    public:
        GeqOp(std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs) : BinaryOp(OpName::GEQ, lhs, rhs, false) {}
    };

    struct MatmulOp : public BinaryOp
    {
    public:
        MatmulOp(std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs) : BinaryOp(OpName::MATMUL, lhs, rhs, false) {}
        void backward(std::shared_ptr<Array> arr) const override;
    };

    struct SqOp : public UnaryOp
    {
    public:
        SqOp(std::shared_ptr<Array> operand, bool in_place) : UnaryOp(OpName::SQ, operand, in_place) {}

        void backward(std::shared_ptr<Array> arr) const override;
    };

    struct SqrtOp : public UnaryOp
    {
    public:
        SqrtOp(std::shared_ptr<Array> operand, bool in_place) : UnaryOp(OpName::SQRT, operand, in_place) {}

        void backward(std::shared_ptr<Array> arr) const override;
    };

    struct NegOp : public UnaryOp
    {
    public:
        NegOp(std::shared_ptr<Array> operand, bool in_place) : UnaryOp(OpName::NEG, operand, in_place) {}

        void backward(std::shared_ptr<Array> arr) const override;
    };

    struct ExpOp : public UnaryOp
    {
    public:
        ExpOp(std::shared_ptr<Array> operand, bool in_place) : UnaryOp(OpName::EXP, operand, in_place) {}

        void backward(std::shared_ptr<Array> arr) const override;
    };

    struct LogOp : public UnaryOp
    {
    public:
        LogOp(std::shared_ptr<Array> operand, bool in_place) : UnaryOp(OpName::LOG, operand, in_place) {}

        void backward(std::shared_ptr<Array> arr) const override;
    };

    struct RecipOp : public UnaryOp
    {
    public:
        RecipOp(std::shared_ptr<Array> operand, bool in_place) : UnaryOp(OpName::RECIP, operand, in_place) {}

        void backward(std::shared_ptr<Array> arr) const override;
    };

    struct ReshapeOp : public TransformOp
    {
    private:
        std::vector<uint64_t> view;

    public:
        ReshapeOp(std::shared_ptr<Array> operand, const std::vector<uint64_t> &view) : TransformOp(OpName::RESHAPE, operand), view(view) {}
        const std::vector<uint64_t> &get_view() { return view; }
        const std::string str() const override { return TransformOp::str() + ", view: (" + vnumstr(view) + ")"; }
        void backward(std::shared_ptr<Array> arr) const override;
    };

    struct SliceOp : public TransformOp
    {
    private:
        std::vector<Range> ranges;

    public:
        SliceOp(std::shared_ptr<Array> operand, const std::vector<Range> &ranges) : TransformOp(OpName::SLICE, operand), ranges(ranges) {}
        const std::vector<Range> &get_ranges() { return ranges; }
        const std::string str() const override
        {
            return TransformOp::str() + ", ranges:(" + vstr<Range>(ranges, [](Range range)
                                                                   { return range.str(); }) +
                   ")";
        }
        void backward(std::shared_ptr<Array> arr) const override;
    };

    struct UnsliceOp : public TransformOp
    {
    private:
        Shape orig_shape;
        std::vector<Range> ranges;

    public:
        UnsliceOp(std::shared_ptr<Array> operand, const Shape &orig_shape, const std::vector<Range> &ranges) : TransformOp(OpName::UNSLICE, operand), orig_shape(orig_shape), ranges(ranges) {}
        const Shape &get_shape() { return orig_shape; }
        const std::vector<Range> &get_ranges() { return ranges; }
        const std::string str() const override
        {
            return TransformOp::str() + ", ranges:(" + vstr<Range>(ranges, [](Range range)
                                                                   { return range.str(); }) +
                   "), original shape: (" + orig_shape.str() + ")";
        }
        void backward(std::shared_ptr<Array> arr) const override;
    };

    struct PermuteOp : public TransformOp
    {
    private:
        std::vector<uint64_t> order;

    public:
        PermuteOp(std::shared_ptr<Array> operand, const std::vector<uint64_t> &order) : TransformOp(OpName::PERMUTE, operand), order(order) {}
        const std::vector<uint64_t> &get_perm() { return order; }
        const std::string str() const override { return TransformOp::str() + ", permutation: (" + vnumstr(order) + ")"; }
        void backward(std::shared_ptr<Array> arr) const override;
    };

    struct BroadcastOp : public TransformOp
    {
    private:
        std::vector<uint64_t> view;

    public:
        BroadcastOp(std::shared_ptr<Array> operand, const std::vector<uint64_t> &view) : TransformOp(OpName::BROADCAST, operand), view(view) {}
        const std::vector<uint64_t> &get_view() { return view; }
        const std::string str() const override
        {
            return TransformOp::str() + ", view: (" + vnumstr(view) + ")";
        }
    };

    struct CopyOp : public Op
    {
    private:
        std::shared_ptr<Array> operand;

    public:
        CopyOp(std::shared_ptr<Array> operand) : Op(OpName::COPY, OpType::MOVE), operand(operand) {}
        std::shared_ptr<Array> get_operand() const { return operand; }
        const std::string str() const override;
    };

    struct SumOp : public ReduceOp
    {
    public:
        SumOp(std::shared_ptr<Array> operand) : ReduceOp(OpName::SUM, operand) {}
    };
}