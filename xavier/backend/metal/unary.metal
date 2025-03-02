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
[[kernel]] void unary_ss(
    device T *input [[buffer(0)]],
    device R *output [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    output[id] = Op()(input[id]);
}

template <class Op, class T, class R>
[[kernel]] void sparse_unary_ss(
    constant const uint *ndim [[buffer(0)]],
    constant const uint *shape [[buffer(1)]],
    constant const uint *stride [[buffer(2)]],
    device T *input [[buffer(3)]],
    device R *output [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    uint idx = access(id, ndim, shape, stride);
    output[id] = Op()(input[idx]);
}

#define unary_float(opname, op) \
template [[host_name(#opname "_f32")]] [[kernel]] decltype(unary_ss<op, float, float>) unary_ss<op, float, float>;                          \
template [[host_name(#opname "_i32")]] [[kernel]] decltype(unary_ss<op, int, float>) unary_ss<op, int, float>;                              \
template [[host_name("sparse_" #opname "_f32")]] [[kernel]] decltype(sparse_unary_ss<op, float, float>) sparse_unary_ss<op, float, float>;  \
template [[host_name("sparse_" #opname "_i32")]] [[kernel]] decltype(sparse_unary_ss<op, int, float>) sparse_unary_ss<op, int, float>;

#define unary_all(opname, op) \
template [[host_name(#opname "_f32")]] [[kernel]] decltype(unary_ss<op, float, float>) unary_ss<op, float, float>;                          \
template [[host_name(#opname "_i32")]] [[kernel]] decltype(unary_ss<op, int, int>) unary_ss<op, int, int>;                                  \
template [[host_name("sparse_" #opname "_f32")]] [[kernel]] decltype(sparse_unary_ss<op, float, float>) sparse_unary_ss<op, float, float>;  \
template [[host_name("sparse_" #opname "_i32")]] [[kernel]] decltype(sparse_unary_ss<op, int, int>) sparse_unary_ss<op, int, int>;

unary_float(exp, Exp)
unary_float(log, Log)
unary_all(neg, Neg)
unary_float(recip, Recip)
unary_all(sq, Sq)
unary_float(sqrt, Sqrt)
