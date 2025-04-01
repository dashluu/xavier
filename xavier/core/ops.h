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
        SUM,
        MAX,
        MIN
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

    inline const std::unordered_map<OpName, const std::string> opnames = {
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
        {OpName::SUM, "sum"},
        {OpName::MAX, "max"},
        {OpName::MIN, "min"}};

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
        const std::string &get_name_str() const { return opnames.at(name); }
        OpType get_type() const { return type; }
        virtual void backward(ArrayPtr arr) const {}
    };

    struct InitializerOp : public Op
    {
    public:
        InitializerOp(OpName name) : Op(name, OpType::INITIALIZER) {}
    };

    struct ArangeOp : public InitializerOp
    {
    private:
        ShapeView view;
        isize start;
        isize step;
        Dtype dtype;

    public:
        ArangeOp(const ShapeView &view, isize start, isize step, const Dtype &dtype) : InitializerOp(OpName::ARANGE), view(view), start(start), step(step), dtype(dtype) {}
        const ShapeView &get_view() { return view; }
        isize get_start() { return start; }
        isize get_step() { return step; }
        const Dtype &get_dtype() { return dtype; }
        const std::string str() const override
        {
            return get_name_str() + ", view: (" + vnumstr(view) + "), start: " + std::to_string(start) + ", step: " + std::to_string(step);
        }
    };

    struct FullOp : public InitializerOp
    {
    private:
        ShapeView view;
        int c;
        Dtype dtype;

    public:
        FullOp(const ShapeView &view, int c, const Dtype &dtype) : InitializerOp(OpName::FULL), view(view), c(c), dtype(dtype) {}
        const ShapeView &get_view() { return view; }
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
        ArrayPtr operand;

    public:
        UnaryOp(OpName name, ArrayPtr operand, bool in_place) : Op(name, OpType::UNARY), operand(operand), in_place(in_place) {}
        ArrayPtr get_operand() const { return operand; }
        const std::string str() const override;
        bool is_in_place() const { return in_place; }
    };

    struct BinaryOp : public Op
    {
    protected:
        bool in_place;
        ArrayPtr lhs;
        ArrayPtr rhs;

    public:
        BinaryOp(OpName name, ArrayPtr lhs, ArrayPtr rhs, bool in_place) : Op(name, OpType::BINARY), lhs(lhs), rhs(rhs), in_place(in_place) {}
        ArrayPtr get_lhs() const { return lhs; }
        ArrayPtr get_rhs() const { return rhs; }
        const std::string str() const override;
        bool is_in_place() const { return in_place; }
    };

    struct TransformOp : public Op
    {
    protected:
        ArrayPtr operand;

    public:
        TransformOp(OpName name, ArrayPtr operand) : Op(name, OpType::TRANSFORM), operand(operand) {}
        ArrayPtr get_operand() const { return operand; }
        const std::string str() const override;
    };

    struct ReduceOp : public Op
    {
    protected:
        ArrayPtr operand;

    public:
        ReduceOp(OpName name, ArrayPtr operand) : Op(name, OpType::REDUCE), operand(operand) {}
        ArrayPtr get_operand() const { return operand; }
        const std::string str() const override;
    };

    struct AddOp : public BinaryOp
    {
    public:
        AddOp(ArrayPtr lhs, ArrayPtr rhs, bool in_place) : BinaryOp(OpName::ADD, lhs, rhs, in_place) {}

        void backward(ArrayPtr arr) const override;
    };

    struct SubOp : public BinaryOp
    {
    public:
        SubOp(ArrayPtr lhs, ArrayPtr rhs, bool in_place) : BinaryOp(OpName::SUB, lhs, rhs, in_place) {}

        void backward(ArrayPtr arr) const override;
    };

    struct MulOp : public BinaryOp
    {
    public:
        MulOp(ArrayPtr lhs, ArrayPtr rhs, bool in_place) : BinaryOp(OpName::MUL, lhs, rhs, in_place) {}

        void backward(ArrayPtr arr) const override;
    };

    struct DivOp : public BinaryOp
    {
    public:
        DivOp(ArrayPtr lhs, ArrayPtr rhs, bool in_place) : BinaryOp(OpName::DIV, lhs, rhs, in_place) {}

        void backward(ArrayPtr arr) const override;
    };

    struct EqOp : public BinaryOp
    {
    public:
        EqOp(ArrayPtr lhs, ArrayPtr rhs) : BinaryOp(OpName::EQ, lhs, rhs, false) {}
    };

    struct NeqOp : public BinaryOp
    {
    public:
        NeqOp(ArrayPtr lhs, ArrayPtr rhs) : BinaryOp(OpName::NEQ, lhs, rhs, false) {}
    };

    struct LtOp : public BinaryOp
    {
    public:
        LtOp(ArrayPtr lhs, ArrayPtr rhs) : BinaryOp(OpName::LT, lhs, rhs, false) {}
    };

    struct GtOp : public BinaryOp
    {
    public:
        GtOp(ArrayPtr lhs, ArrayPtr rhs) : BinaryOp(OpName::GT, lhs, rhs, false) {}
    };

    struct LeqOp : public BinaryOp
    {
    public:
        LeqOp(ArrayPtr lhs, ArrayPtr rhs) : BinaryOp(OpName::LEQ, lhs, rhs, false) {}
    };

    struct GeqOp : public BinaryOp
    {
    public:
        GeqOp(ArrayPtr lhs, ArrayPtr rhs) : BinaryOp(OpName::GEQ, lhs, rhs, false) {}
    };

    struct MatmulOp : public BinaryOp
    {
    public:
        MatmulOp(ArrayPtr lhs, ArrayPtr rhs) : BinaryOp(OpName::MATMUL, lhs, rhs, false) {}
        void backward(ArrayPtr arr) const override;
    };

    struct SqOp : public UnaryOp
    {
    public:
        SqOp(ArrayPtr operand, bool in_place) : UnaryOp(OpName::SQ, operand, in_place) {}

        void backward(ArrayPtr arr) const override;
    };

    struct SqrtOp : public UnaryOp
    {
    public:
        SqrtOp(ArrayPtr operand, bool in_place) : UnaryOp(OpName::SQRT, operand, in_place) {}

        void backward(ArrayPtr arr) const override;
    };

    struct NegOp : public UnaryOp
    {
    public:
        NegOp(ArrayPtr operand, bool in_place) : UnaryOp(OpName::NEG, operand, in_place) {}

        void backward(ArrayPtr arr) const override;
    };

    struct ExpOp : public UnaryOp
    {
    public:
        ExpOp(ArrayPtr operand, bool in_place) : UnaryOp(OpName::EXP, operand, in_place) {}

        void backward(ArrayPtr arr) const override;
    };

    struct LogOp : public UnaryOp
    {
    public:
        LogOp(ArrayPtr operand, bool in_place) : UnaryOp(OpName::LOG, operand, in_place) {}

        void backward(ArrayPtr arr) const override;
    };

    struct RecipOp : public UnaryOp
    {
    public:
        RecipOp(ArrayPtr operand, bool in_place) : UnaryOp(OpName::RECIP, operand, in_place) {}

        void backward(ArrayPtr arr) const override;
    };

    struct ReshapeOp : public TransformOp
    {
    private:
        ShapeView view;

    public:
        ReshapeOp(ArrayPtr operand, const ShapeView &view) : TransformOp(OpName::RESHAPE, operand), view(view) {}
        const ShapeView &get_view() { return view; }
        const std::string str() const override { return TransformOp::str() + ", view: (" + vnumstr(view) + ")"; }
        void backward(ArrayPtr arr) const override;
    };

    struct SliceOp : public TransformOp
    {
    private:
        std::vector<Range> ranges;

    public:
        SliceOp(ArrayPtr operand, const std::vector<Range> &ranges) : TransformOp(OpName::SLICE, operand), ranges(ranges) {}
        const std::vector<Range> &get_ranges() { return ranges; }
        const std::string str() const override
        {
            return TransformOp::str() + ", ranges:(" + vstr<Range>(ranges, [](Range range)
                                                                   { return range.str(); }) +
                   ")";
        }
        void backward(ArrayPtr arr) const override;
    };

    struct UnsliceOp : public TransformOp
    {
    private:
        Shape orig_shape;
        std::vector<Range> ranges;

    public:
        UnsliceOp(ArrayPtr operand, const Shape &orig_shape, const std::vector<Range> &ranges) : TransformOp(OpName::UNSLICE, operand), orig_shape(orig_shape), ranges(ranges) {}
        const Shape &get_shape() { return orig_shape; }
        const std::vector<Range> &get_ranges() { return ranges; }
        const std::string str() const override
        {
            return TransformOp::str() + ", ranges:(" + vstr<Range>(ranges, [](Range range)
                                                                   { return range.str(); }) +
                   "), original shape: (" + orig_shape.str() + ")";
        }
        void backward(ArrayPtr arr) const override;
    };

    struct PermuteOp : public TransformOp
    {
    private:
        ShapeOrder order;

    public:
        PermuteOp(ArrayPtr operand, const ShapeOrder &order) : TransformOp(OpName::PERMUTE, operand), order(order) {}
        const ShapeOrder &get_perm() { return order; }
        const std::string str() const override { return TransformOp::str() + ", permutation: (" + vnumstr(order) + ")"; }
        void backward(ArrayPtr arr) const override;
    };

    struct BroadcastOp : public TransformOp
    {
    private:
        ShapeView view;

    public:
        BroadcastOp(ArrayPtr operand, const ShapeView &view) : TransformOp(OpName::BROADCAST, operand), view(view) {}
        const ShapeView &get_view() { return view; }
        const std::string str() const override
        {
            return TransformOp::str() + ", view: (" + vnumstr(view) + ")";
        }
    };

    struct CopyOp : public Op
    {
    private:
        ArrayPtr operand;

    public:
        CopyOp(ArrayPtr operand) : Op(OpName::COPY, OpType::MOVE), operand(operand) {}
        ArrayPtr get_operand() const { return operand; }
        const std::string str() const override;
    };

    struct SumOp : public ReduceOp
    {
    public:
        SumOp(ArrayPtr operand) : ReduceOp(OpName::SUM, operand) {}
        void backward(ArrayPtr arr) const override;
    };

    struct MaxOp : public ReduceOp
    {
    public:
        MaxOp(ArrayPtr operand) : ReduceOp(OpName::MAX, operand) {}
    };

    struct MinOp : public ReduceOp
    {
    public:
        MinOp(ArrayPtr operand) : ReduceOp(OpName::MIN, operand) {}
    };
}