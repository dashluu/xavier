#include "utils.h"

struct Identity
{
    template <typename T>
    float operator()(T x) const
    {
        return x;
    }
};

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
kernel void unary_ss_vv(
    constant const uint *offset [[buffer(0)]],
    device T *input [[buffer(1)]],
    device R *output [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    output[offset[1] + id] = Op()(input[offset[0] + id]);
}

template <class Op, class T, class R>
kernel void unary_ss_sv(
    constant const uint *ndim [[buffer(0)]],
    constant const uint *offset [[buffer(1)]],
    constant const uint *shape [[buffer(2)]],
    constant const int *output_stride [[buffer(3)]],
    device T *input [[buffer(4)]],
    device R *output [[buffer(5)]],
    uint id [[thread_position_in_grid]])
{
    uint output_idx = strided_idx(id, ndim, shape, output_stride);
    output[offset[1] + output_idx] = Op()(input[offset[0] + id]);
}

template <class Op, class T, class R>
kernel void unary_ss_vs(
    constant const uint *ndim [[buffer(0)]],
    constant const uint *offset [[buffer(1)]],
    constant const uint *shape [[buffer(2)]],
    constant const int *input_stride [[buffer(3)]],
    device T *input [[buffer(4)]],
    device R *output [[buffer(5)]],
    uint id [[thread_position_in_grid]])
{
    uint input_idx = strided_idx(id, ndim, shape, input_stride);
    output[offset[1] + id] = Op()(input[offset[0] + input_idx]);
}

template <class Op, class T, class R>
kernel void unary_ss_ss(
    constant const uint *ndim [[buffer(0)]],
    constant const uint *offset [[buffer(1)]],
    constant const uint *shape [[buffer(2)]],
    constant const int *input_stride [[buffer(3)]],
    constant const int *output_stride [[buffer(4)]],
    device T *input [[buffer(5)]],
    device R *output [[buffer(6)]],
    uint id [[thread_position_in_grid]])
{
    uint input_idx = strided_idx(id, ndim, shape, input_stride);
    uint output_idx = strided_idx(id, ndim, shape, output_stride);
    output[offset[1] + output_idx] = Op()(input[offset[0] + input_idx]);
}

#define unary_float(opname, op) \
template [[host_name(#opname "_vv_f32")]] [[kernel]] decltype(unary_ss_vv<op, float, float>) unary_ss_vv<op, float, float>; \
template [[host_name(#opname "_vv_i32")]] [[kernel]] decltype(unary_ss_vv<op, int, float>) unary_ss_vv<op, int, float>;     \
template [[host_name(#opname "_sv_f32")]] [[kernel]] decltype(unary_ss_sv<op, float, float>) unary_ss_sv<op, float, float>; \
template [[host_name(#opname "_sv_i32")]] [[kernel]] decltype(unary_ss_sv<op, int, float>) unary_ss_sv<op, int, float>;     \
template [[host_name(#opname "_vs_f32")]] [[kernel]] decltype(unary_ss_vs<op, float, float>) unary_ss_vs<op, float, float>; \
template [[host_name(#opname "_vs_i32")]] [[kernel]] decltype(unary_ss_vs<op, int, float>) unary_ss_vs<op, int, float>;     \
template [[host_name(#opname "_ss_f32")]] [[kernel]] decltype(unary_ss_ss<op, float, float>) unary_ss_ss<op, float, float>; \
template [[host_name(#opname "_ss_i32")]] [[kernel]] decltype(unary_ss_ss<op, int, float>) unary_ss_ss<op, int, float>;

#define unary_all(opname, op) \
template [[host_name(#opname "_vv_f32")]] [[kernel]] decltype(unary_ss_vv<op, float, float>) unary_ss_vv<op, float, float>; \
template [[host_name(#opname "_vv_i32")]] [[kernel]] decltype(unary_ss_vv<op, int, int>) unary_ss_vv<op, int, int>;         \
template [[host_name(#opname "_sv_f32")]] [[kernel]] decltype(unary_ss_sv<op, float, float>) unary_ss_sv<op, float, float>; \
template [[host_name(#opname "_sv_i32")]] [[kernel]] decltype(unary_ss_sv<op, int, int>) unary_ss_sv<op, int, int>;         \
template [[host_name(#opname "_vs_f32")]] [[kernel]] decltype(unary_ss_vs<op, float, float>) unary_ss_vs<op, float, float>; \
template [[host_name(#opname "_vs_i32")]] [[kernel]] decltype(unary_ss_vs<op, int, int>) unary_ss_vs<op, int, int>;         \
template [[host_name(#opname "_ss_f32")]] [[kernel]] decltype(unary_ss_ss<op, float, float>) unary_ss_ss<op, float, float>; \
template [[host_name(#opname "_ss_i32")]] [[kernel]] decltype(unary_ss_ss<op, int, int>) unary_ss_ss<op, int, int>;

unary_all(identity, Identity)
unary_all(exp, Exp)
unary_float(log, Log)
unary_all(neg, Neg)
unary_float(recip, Recip)
unary_all(sq, Sq)
unary_float(sqrt, Sqrt)
