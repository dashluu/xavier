#include <metal_stdlib>

struct Exp
{
    template <typename T>
    float operator()(T x) const
    {
        return metal::exp(static_cast<float>(x));
    }
};

struct Log
{
    template <typename T>
    float operator()(T x) const
    {
        return metal::log(static_cast<float>(x));
    }
};

struct Neg
{
    template <typename T>
    T operator()(T x) const
    {
        return -x;
    }
};

struct Recip
{
    template <typename T>
    float operator()(T x) const
    {
        return 1.0f / x;
    }
};

// Unary operations for scalar-scalar
template <class Op, class T, class R>
[[kernel]] void unary_ss(
    device T *input [[buffer(0)]],
    device R *output [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    output[id] = Op()(input[id]);
}

#define unary_float(opname, op) \
template [[host_name(#opname "_f32")]] [[kernel]] decltype(unary_ss<op, float, float>) unary_ss<op, float, float>; \
template [[host_name(#opname "_i32")]] [[kernel]] decltype(unary_ss<op, int, float>) unary_ss<op, int, float>;

#define unary_all(opname, op) \
template [[host_name(#opname "_f32")]] [[kernel]] decltype(unary_ss<op, float, float>) unary_ss<op, float, float>; \
template [[host_name(#opname "_i32")]] [[kernel]] decltype(unary_ss<op, int, int>) unary_ss<op, int, int>;

unary_float(exp, Exp)
unary_float(log, Log)
unary_all(neg, Neg)
unary_float(recip, Recip)
