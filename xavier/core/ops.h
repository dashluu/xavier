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
        IADD,
        SUB,
        ISUB,
        MUL,
        IMUL,
        DIV,
        IDIV,
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
        SLICE
    };

    enum class OpType
    {
        INITIALIZER,
        UNARY,
        BINARY,
        IBINARY,
        TRANSFORM,
        REDUCE
    };

    inline const std::unordered_map<OpName, std::string> opnames = {
        {OpName::RANDN, "randn"},
        {OpName::ARANGE, "arange"},
        {OpName::FULL, "full"},
        {OpName::BUFF, "buff"},
        {OpName::ADD, "add"},
        {OpName::IADD, "iadd"},
        {OpName::SUB, "sub"},
        {OpName::ISUB, "isub"},
        {OpName::MUL, "mul"},
        {OpName::IMUL, "imul"},
        {OpName::DIV, "div"},
        {OpName::IDIV, "idiv"},
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
    };

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
        OpType get_type() const { return type; }
        virtual void backward(std::shared_ptr<Array> arr) const {}
    };

    struct InitializerOp : public Op
    {
    public:
        InitializerOp(OpName name) : Op(name, OpType::INITIALIZER) {}
        const std::string str() const override { return opnames.at(name); }
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
            return InitializerOp::str() + ", view: (" + numstr(view) + "), start: " + std::to_string(start) + ", step: " + std::to_string(step);
        }
    };

    struct FullOp : public InitializerOp
    {
    private:
        std::vector<uint64_t> view;
        float c;
        Dtype dtype;

    public:
        FullOp(const std::vector<uint64_t> &view, float c, const Dtype &dtype) : InitializerOp(OpName::FULL), view(view), c(c), dtype(dtype) {}
        const std::vector<uint64_t> &get_view() { return view; }
        float get_const() const { return c; }
        const Dtype &get_dtype() { return dtype; }
        const std::string str() const override { return InitializerOp::str() + ", view: (" + numstr(view) + "), value: " + std::to_string(c); }
    };

    struct BuffOp : public InitializerOp
    {
    public:
        BuffOp() : InitializerOp(OpName::BUFF) {}
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
        void backward_helper(std::shared_ptr<Array> arr, std::shared_ptr<Array> lgrad, std::shared_ptr<Array> rgrad) const;
    };

    struct IBinaryOp : public RootBinaryOp
    {
    public:
        IBinaryOp(OpName name, std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs) : RootBinaryOp(name, OpType::IBINARY, lhs, rhs) {}
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

    struct IAddOp : public IBinaryOp
    {
    public:
        IAddOp(std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs) : IBinaryOp(OpName::IADD, lhs, rhs) {}
    };

    struct SubOp : public BinaryOp
    {
    public:
        SubOp(std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs) : BinaryOp(OpName::SUB, lhs, rhs) {}

        void backward(std::shared_ptr<Array> arr) const override;
    };

    struct ISubOp : public IBinaryOp
    {
    public:
        ISubOp(std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs) : IBinaryOp(OpName::ISUB, lhs, rhs) {}
    };

    struct MulOp : public BinaryOp
    {
    public:
        MulOp(std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs) : BinaryOp(OpName::MUL, lhs, rhs) {}

        void backward(std::shared_ptr<Array> arr) const override;
    };

    struct IMulOp : public IBinaryOp
    {
    public:
        IMulOp(std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs) : IBinaryOp(OpName::IMUL, lhs, rhs) {}
    };

    struct DivOp : public BinaryOp
    {
    public:
        DivOp(std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs) : BinaryOp(OpName::DIV, lhs, rhs) {}

        void backward(std::shared_ptr<Array> arr) const override;
    };

    struct IDivOp : public IBinaryOp
    {
    public:
        IDivOp(std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs) : IBinaryOp(OpName::IDIV, lhs, rhs) {}
    };

    struct SqOp : public UnaryOp
    {
    public:
        SqOp(std::shared_ptr<Array> operand) : UnaryOp(OpName::SQ, operand) {}
    };

    struct SqrtOp : public UnaryOp
    {
    public:
        SqrtOp(std::shared_ptr<Array> operand) : UnaryOp(OpName::SQRT, operand) {}
    };

    struct NegOp : public UnaryOp
    {
    public:
        NegOp(std::shared_ptr<Array> operand) : UnaryOp(OpName::NEG, operand) {}
    };

    struct ExpOp : public UnaryOp
    {
    public:
        ExpOp(std::shared_ptr<Array> operand) : UnaryOp(OpName::EXP, operand) {}
    };

    struct LogOp : public UnaryOp
    {
    public:
        LogOp(std::shared_ptr<Array> operand) : UnaryOp(OpName::LOG, operand) {}
    };

    struct RecipOp : public UnaryOp
    {
    public:
        RecipOp(std::shared_ptr<Array> operand) : UnaryOp(OpName::RECIP, operand) {}
    };

    struct ReshapeOp : public TransformOp
    {
    private:
        std::vector<uint64_t> view;
        bool copy;

    public:
        ReshapeOp(std::shared_ptr<Array> operand, const std::vector<uint64_t> &view, bool copy) : TransformOp(OpName::RESHAPE, operand), view(view), copy(copy) {}
        const std::vector<uint64_t> &get_view() { return view; }
        bool get_copy() { return copy; }
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
        BroadcastOp(std::shared_ptr<Array> operand, const std::vector<uint64_t> &view) : TransformOp(OpName::SLICE, operand), view(view) {}
        const std::vector<uint64_t> &get_view() { return view; }
        const std::string str() const override
        {
            return TransformOp::str() + ", view: (" + numstr(view) + ")";
        }
    };
}