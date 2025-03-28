#include "utils.h"

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

struct Sqrt
{
    template <typename T>
    float operator()(T x) const
    {
        return metal::sqrt(static_cast<float>(x));
    }
};

struct Sq
{
    template <typename T>
    float operator()(T x) const
    {
        return x * x;
    }
};

// Unary operations for scalar-scalar
template <class Op, class T, class R>
kernel void unary_ss(
    constant const uint *offset [[buffer(0)]],
    device T *input [[buffer(1)]],
    device R *output [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    output[offset[1] + id] = Op()(input[offset[0] + id]);
}

template <class Op, class T, class R>
kernel void strided_unary_ss(
    constant const uint *ndim [[buffer(0)]],
    constant const uint *offset [[buffer(1)]],
    constant const uint *shape [[buffer(2)]],
    constant const int *stride [[buffer(3)]],
    device T *input [[buffer(4)]],
    device R *output [[buffer(5)]],
    uint id [[thread_position_in_grid]])
{
    uint idx = strided_idx(id, ndim, shape, stride);
    output[offset[1] + id] = Op()(input[offset[0] + idx]);
}

#define unary_float(opname, op) \
template [[host_name(#opname "_f32")]] [[kernel]] decltype(unary_ss<op, float, float>) unary_ss<op, float, float>;                              \
template [[host_name(#opname "_i32")]] [[kernel]] decltype(unary_ss<op, int, float>) unary_ss<op, int, float>;                                  \
template [[host_name("strided_" #opname "_f32")]] [[kernel]] decltype(strided_unary_ss<op, float, float>) strided_unary_ss<op, float, float>;   \
template [[host_name("strided_" #opname "_i32")]] [[kernel]] decltype(strided_unary_ss<op, int, float>) strided_unary_ss<op, int, float>;

#define unary_all(opname, op) \
template [[host_name(#opname "_f32")]] [[kernel]] decltype(unary_ss<op, float, float>) unary_ss<op, float, float>;                              \
template [[host_name(#opname "_i32")]] [[kernel]] decltype(unary_ss<op, int, int>) unary_ss<op, int, int>;                                      \
template [[host_name("strided_" #opname "_f32")]] [[kernel]] decltype(strided_unary_ss<op, float, float>) strided_unary_ss<op, float, float>;   \
template [[host_name("strided_" #opname "_i32")]] [[kernel]] decltype(strided_unary_ss<op, int, int>) strided_unary_ss<op, int, int>;

unary_all(exp, Exp)
unary_float(log, Log)
unary_all(neg, Neg)
unary_float(recip, Recip)
unary_all(sq, Sq)
unary_float(sqrt, Sqrt)
