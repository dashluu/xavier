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
        ADD,
        SELF_ADD,
        SUB,
        SELF_SUB,
        MUL,
        SELF_MUL,
        DIV,
        SELF_DIV,
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
        TRANSPOSE,
        BROADCAST,
        SQUEEZE,
        UNSQUEEZE,
        INTERPRET,
        SLICE,
        MOVE
    };

    enum class OpType
    {
        INITIALIZER,
        UNARY,
        SELF_UNARY,
        BINARY,
        SELF_BINARY,
        TRANSFORM,
        REDUCE,
        MOVE
    };

    inline const std::unordered_map<OpName, std::string> opnames = {
        {OpName::RANDN, "randn"},
        {OpName::ARANGE, "arange"},
        {OpName::FULL, "full"},
        {OpName::BUFF, "buff"},
        {OpName::ADD, "add"},
        {OpName::SELF_ADD, "self_add"},
        {OpName::SUB, "sub"},
        {OpName::SELF_SUB, "self_sub"},
        {OpName::MUL, "mul"},
        {OpName::SELF_MUL, "self_mul"},
        {OpName::DIV, "div"},
        {OpName::SELF_DIV, "self_div"},
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
        {OpName::TRANSPOSE, "transpose"},
        {OpName::INTERPRET, "interpret"},
        {OpName::SLICE, "slice"},
        {OpName::MOVE, "move"}};

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
            return get_name_str() + ", view: (" + numstr(view) + "), start: " + std::to_string(start) + ", step: " + std::to_string(step);
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
            auto s = get_name_str() + ", view: (" + numstr(view) + "), value: ";
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

    struct UnaryOp : public Op
    {
    protected:
        std::shared_ptr<Array> operand;

    public:
        UnaryOp(OpName name, std::shared_ptr<Array> operand) : Op(name, OpType::UNARY), operand(operand) {}
        std::shared_ptr<Array> get_operand() const { return operand; }
        const std::string str() const override;
    };

    struct RootBinaryOp : public Op
    {
    protected:
        std::shared_ptr<Array> lhs;
        std::shared_ptr<Array> rhs;

    public:
        RootBinaryOp(OpName name, OpType type, std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs) : Op(name, type), lhs(lhs), rhs(rhs) {}
        std::shared_ptr<Array> get_lhs() const { return lhs; }
        std::shared_ptr<Array> get_rhs() const { return rhs; }
        const std::string str() const override;
    };

    struct BinaryOp : public RootBinaryOp
    {
    public:
        BinaryOp(OpName name, std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs) : RootBinaryOp(name, OpType::BINARY, lhs, rhs) {}
    };

    struct SelfBinaryOp : public RootBinaryOp
    {
    public:
        SelfBinaryOp(OpName name, std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs) : RootBinaryOp(name, OpType::SELF_BINARY, lhs, rhs) {}
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

    struct AddOp : public BinaryOp
    {
    public:
        AddOp(std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs) : BinaryOp(OpName::ADD, lhs, rhs) {}

        void backward(std::shared_ptr<Array> arr) const override;
    };

    struct SelfAddOp : public SelfBinaryOp
    {
    public:
        SelfAddOp(std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs) : SelfBinaryOp(OpName::SELF_ADD, lhs, rhs) {}

        void backward(std::shared_ptr<Array> arr) const override;
    };

    struct SubOp : public BinaryOp
    {
    public:
        SubOp(std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs) : BinaryOp(OpName::SUB, lhs, rhs) {}

        void backward(std::shared_ptr<Array> arr) const override;
    };

    struct SelfSubOp : public SelfBinaryOp
    {
    public:
        SelfSubOp(std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs) : SelfBinaryOp(OpName::SELF_SUB, lhs, rhs) {}
    };

    struct MulOp : public BinaryOp
    {
    public:
        MulOp(std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs) : BinaryOp(OpName::MUL, lhs, rhs) {}

        void backward(std::shared_ptr<Array> arr) const override;
    };

    struct SelfMulOp : public SelfBinaryOp
    {
    public:
        SelfMulOp(std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs) : SelfBinaryOp(OpName::SELF_MUL, lhs, rhs) {}
    };

    struct DivOp : public BinaryOp
    {
    public:
        DivOp(std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs) : BinaryOp(OpName::DIV, lhs, rhs) {}

        void backward(std::shared_ptr<Array> arr) const override;
    };

    struct SelfDivOp : public SelfBinaryOp
    {
    public:
        SelfDivOp(std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs) : SelfBinaryOp(OpName::SELF_DIV, lhs, rhs) {}
    };

    struct EqOp : public BinaryOp
    {
    public:
        EqOp(std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs) : BinaryOp(OpName::EQ, lhs, rhs) {}
    };

    struct NeqOp : public BinaryOp
    {
    public:
        NeqOp(std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs) : BinaryOp(OpName::NEQ, lhs, rhs) {}
    };

    struct LtOp : public BinaryOp
    {
    public:
        LtOp(std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs) : BinaryOp(OpName::LT, lhs, rhs) {}
    };

    struct GtOp : public BinaryOp
    {
    public:
        GtOp(std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs) : BinaryOp(OpName::GT, lhs, rhs) {}
    };

    struct LeqOp : public BinaryOp
    {
    public:
        LeqOp(std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs) : BinaryOp(OpName::LEQ, lhs, rhs) {}
    };

    struct GeqOp : public BinaryOp
    {
    public:
        GeqOp(std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs) : BinaryOp(OpName::GEQ, lhs, rhs) {}
    };

    struct MatmulOp : public BinaryOp
    {
    public:
        MatmulOp(std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs) : BinaryOp(OpName::MATMUL, lhs, rhs) {}
        void backward(std::shared_ptr<Array> arr) const override;
    };

    struct SqOp : public UnaryOp
    {
    public:
        SqOp(std::shared_ptr<Array> operand) : UnaryOp(OpName::SQ, operand) {}

        void backward(std::shared_ptr<Array> arr) const override;
    };

    struct SqrtOp : public UnaryOp
    {
    public:
        SqrtOp(std::shared_ptr<Array> operand) : UnaryOp(OpName::SQRT, operand) {}

        void backward(std::shared_ptr<Array> arr) const override;
    };

    struct NegOp : public UnaryOp
    {
    public:
        NegOp(std::shared_ptr<Array> operand) : UnaryOp(OpName::NEG, operand) {}

        void backward(std::shared_ptr<Array> arr) const override;
    };

    struct ExpOp : public UnaryOp
    {
    public:
        ExpOp(std::shared_ptr<Array> operand) : UnaryOp(OpName::EXP, operand) {}

        void backward(std::shared_ptr<Array> arr) const override;
    };

    struct LogOp : public UnaryOp
    {
    public:
        LogOp(std::shared_ptr<Array> operand) : UnaryOp(OpName::LOG, operand) {}

        void backward(std::shared_ptr<Array> arr) const override;
    };

    struct RecipOp : public UnaryOp
    {
    public:
        RecipOp(std::shared_ptr<Array> operand) : UnaryOp(OpName::RECIP, operand) {}

        void backward(std::shared_ptr<Array> arr) const override;
    };

    struct ReshapeOp : public TransformOp
    {
    private:
        std::vector<uint64_t> view;

    public:
        ReshapeOp(std::shared_ptr<Array> operand, const std::vector<uint64_t> &view) : TransformOp(OpName::RESHAPE, operand), view(view) {}
        const std::vector<uint64_t> &get_view() { return view; }
        const std::string str() const override { return TransformOp::str() + ", view: (" + numstr(view) + ")"; }
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
            return TransformOp::str() + ", view: (" + numstr(view) + ")";
        }
    };

    struct MoveOp : public Op
    {
    private:
        std::shared_ptr<Array> operand;

    public:
        MoveOp(std::shared_ptr<Array> operand) : Op(OpName::MOVE, OpType::MOVE), operand(operand) {}
        std::shared_ptr<Array> get_operand() const { return operand; }
        const std::string str() const override;
    };
}