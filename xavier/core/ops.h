#pragma once

#include "../common.h"
#include "dtype.h"

namespace xv::core
{
    enum class OpName
    {
        RANDN,
        ARANGE,
        CONSTANT,
        BUFF,
        ADD,
        SUB,
        MUL,
        DIV,
        MATMUL,
        NEG,
        EXP,
        LOG,
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
        TRANSFORM,
        REDUCE
    };

    inline const std::unordered_map<OpName, std::string> opnames = {
        {OpName::RANDN, "randn"},
        {OpName::ARANGE, "arange"},
        {OpName::CONSTANT, "constant"},
        {OpName::BUFF, "buff"},
        {OpName::ADD, "add"},
        {OpName::SUB, "sub"},
        {OpName::MUL, "mul"},
        {OpName::DIV, "div"},
        {OpName::MATMUL, "matmul"},
        {OpName::NEG, "neg"},
        {OpName::EXP, "exp"},
        {OpName::LOG, "log"},
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
    };

    struct ArangeOp : public Op
    {
    private:
        std::vector<uint64_t> view;
        int64_t start;
        int64_t step;
        Dtype dtype;

    public:
        ArangeOp(const std::vector<uint64_t> &view, int64_t start, int64_t step, const Dtype &dtype) : Op(OpName::ARANGE, OpType::INITIALIZER), view(view), start(start), step(step), dtype(dtype) {}
        const std::vector<uint64_t> &get_view() { return view; }
        int64_t get_start() { return start; }
        int64_t get_step() { return step; }
        const Dtype &get_dtype() { return dtype; }
        const std::string str() const override { return opnames.at(name) + "((" + numstr(view) + "), " + std::to_string(start) + ", " + std::to_string(step) + ")"; }
    };

    struct ConstOp : public Op
    {
    private:
        std::vector<uint64_t> view;
        float c;
        Dtype dtype;

    public:
        ConstOp(const std::vector<uint64_t> &view, float c, const Dtype &dtype) : Op(OpName::CONSTANT, OpType::INITIALIZER), view(view), c(c), dtype(dtype) {}
        const std::vector<uint64_t> &get_view() { return view; }
        float get_const() const { return c; }
        const Dtype &get_dtype() { return dtype; }
        const std::string str() const override { return opnames.at(name) + "((" + numstr(view) + "), " + std::to_string(c) + ")"; }
    };

    struct BuffOp : public Op
    {
    public:
        BuffOp() : Op(OpName::BUFF, OpType::INITIALIZER) {}
        const std::string str() const override { return opnames.at(name); }
    };

    struct UnaryOp : public Op
    {
    protected:
        std::shared_ptr<Array> operand;

    public:
        UnaryOp(OpName name, std::shared_ptr<Array> operand) : Op(name, OpType::UNARY), operand(operand) {}
        std::shared_ptr<Array> get_operand() const { return operand; }
        const std::string str() const override { return opnames.at(name); }
    };

    struct BinaryOp : public Op
    {
    protected:
        std::shared_ptr<Array> lhs;
        std::shared_ptr<Array> rhs;

    public:
        BinaryOp(OpName name, std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs) : Op(name, OpType::BINARY), lhs(lhs), rhs(rhs) {}
        std::shared_ptr<Array> get_lhs() const { return lhs; }
        std::shared_ptr<Array> get_rhs() const { return rhs; }
        const std::string str() const override { return opnames.at(name); }
    };

    struct TransformOp : public Op
    {
    protected:
        std::shared_ptr<Array> operand;

    public:
        TransformOp(OpName name, std::shared_ptr<Array> operand) : Op(name, OpType::TRANSFORM), operand(operand) {}
        std::shared_ptr<Array> get_operand() const { return operand; }
    };

    struct AddOp : public BinaryOp
    {
    public:
        AddOp(std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs) : BinaryOp(OpName::ADD, lhs, rhs) {}
    };

    struct SubOp : public BinaryOp
    {
    public:
        SubOp(std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs) : BinaryOp(OpName::SUB, lhs, rhs) {}
    };

    struct MulOp : public BinaryOp
    {
    public:
        MulOp(std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs) : BinaryOp(OpName::MUL, lhs, rhs) {}
    };

    struct DivOp : public BinaryOp
    {
    public:
        DivOp(std::shared_ptr<Array> lhs, std::shared_ptr<Array> rhs) : BinaryOp(OpName::DIV, lhs, rhs) {}
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

    struct ReshapeOp : public TransformOp
    {
    private:
        std::vector<uint64_t> view;
        bool copy;

    public:
        ReshapeOp(std::shared_ptr<Array> operand, const std::vector<uint64_t> &view, bool copy) : TransformOp(OpName::RESHAPE, operand), view(view), copy(copy) {}
        const std::vector<uint64_t> &get_view() { return view; }
        bool get_copy() { return copy; }
        const std::string str() const override { return opnames.at(name) + "(" + numstr(view) + ")"; }
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
            return opnames.at(name) + "(" + vstr<Range>(ranges, [](Range range)
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
            return opnames.at(name) + "(" + numstr(view) + ")";
        }
    };
}