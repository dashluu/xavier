#include "utils.h"

struct Add
{
    template <class T>
    T operator()(T lhs, T rhs) { return lhs + rhs; }
};

struct Sub
{
    template <class T>
    T operator()(T lhs, T rhs) { return lhs - rhs; }
};

struct Mul
{
    template <class T>
    T operator()(T lhs, T rhs) { return lhs * rhs; }
};

struct Div
{
    template <class T>
    T operator()(T lhs, T rhs) { return lhs / rhs; }
};

struct Eq
{
    template <class T>
    bool operator()(T lhs, T rhs) { return lhs == rhs; }
};

struct Neq
{
    template <class T>
    bool operator()(T lhs, T rhs) { return lhs != rhs; }
};

struct Lt
{
    template <class T>
    bool operator()(T lhs, T rhs) { return lhs < rhs; }
};

struct Gt
{
    template <class T>
    bool operator()(T lhs, T rhs) { return lhs > rhs; }
};

struct Leq
{
    template <class T>
    bool operator()(T lhs, T rhs) { return lhs <= rhs; }
};

struct Geq
{
    template <class T>
    bool operator()(T lhs, T rhs) { return lhs >= rhs; }
};

// Binary operations for scalar-scalar
template <class Op, class T, class R>
kernel void binary_ss_vv(
    constant const uint *offset [[buffer(0)]],
    device T *lhs [[buffer(1)]],
    device T *rhs [[buffer(2)]],
    device R *output [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    output[offset[2] + id] = Op()(lhs[offset[0] + id], rhs[offset[1] + id]);
}

template <class Op, class T, class R>
kernel void binary_ss_sv(
    constant const uint *ndim [[buffer(0)]],
    constant const uint *offset [[buffer(1)]],
    constant const uint *shape [[buffer(2)]],
    onstant const int *output_stride [[buffer(3)]],
    device T *lhs [[buffer(4)]],
    device T *rhs [[buffer(5)]],
    device R *output [[buffer(6)]],
    uint id [[thread_position_in_grid]])
{
    uint output_idx = strided_idx(id, ndim, shape, output_stride);
    output[offset[2] + output_idx] = Op()(lhs[offset[0] + id], rhs[offset[1] + id]);
}

template <class Op, class T, class R>
kernel void binary_ss_vs(
    constant const uint *ndim [[buffer(0)]],
    constant const uint *offset [[buffer(1)]],
    constant const uint *shape [[buffer(2)]],
    constant const int *lhs_stride [[buffer(3)]],
    constant const int *rhs_stride [[buffer(4)]],
    device T *lhs [[buffer(5)]],
    device T *rhs [[buffer(6)]],
    device R *output [[buffer(7)]],
    uint id [[thread_position_in_grid]])
{
    uint lhs_idx = strided_idx(id, ndim, shape, lhs_stride);
    uint rhs_idx = strided_idx(id, ndim, shape, rhs_stride);
    output[offset[2] + id] = Op()(lhs[offset[0] + lhs_idx], rhs[offset[1] + rhs_idx]);
}

template <class Op, class T, class R>
kernel void binary_ss_ss(
    constant const uint *ndim [[buffer(0)]],
    constant const uint *offset [[buffer(1)]],
    constant const uint *shape [[buffer(2)]],
    constant const int *lhs_stride [[buffer(3)]],
    constant const int *rhs_stride [[buffer(4)]],
    constant const int *output_stride [[buffer(5)]],
    device T *lhs [[buffer(6)]],
    device T *rhs [[buffer(7)]],
    device R *output [[buffer(8)]],
    uint id [[thread_position_in_grid]])
{
    uint lhs_idx = strided_idx(id, ndim, shape, lhs_stride);
    uint rhs_idx = strided_idx(id, ndim, shape, rhs_stride);
    uint output_idx = strided_idx(id, ndim, shape, output_stride);
    output[offset[2] + output_idx] = Op()(lhs[offset[0] + lhs_idx], rhs[offset[1] + rhs_idx]);
}

#define numeric_binary(opname, op) \
template [[host_name(#opname "_vv_f32")]] [[kernel]] decltype(binary_ss_vv<op, float, float>) binary_ss_vv<op, float, float>;   \
template [[host_name(#opname "_vv_i32")]] [[kernel]] decltype(binary_ss_vv<op, int, int>) binary_ss_vv<op, int, int>;           \
template [[host_name(#opname "_sv_f32")]] [[kernel]] decltype(binary_ss_sv<op, float, float>) binary_ss_sv<op, float, float>;   \
template [[host_name(#opname "_sv_i32")]] [[kernel]] decltype(binary_ss_sv<op, int, int>) binary_ss_sv<op, int, int>;           \
template [[host_name(#opname "_vs_f32")]] [[kernel]] decltype(binary_ss_vs<op, float, float>) binary_ss_vs<op, float, float>;   \
template [[host_name(#opname "_vs_i32")]] [[kernel]] decltype(binary_ss_vs<op, int, int>) binary_ss_vs<op, int, int>;           \
template [[host_name(#opname "_ss_f32")]] [[kernel]] decltype(binary_ss_ss<op, float, float>) binary_ss_ss<op, float, float>;   \
template [[host_name(#opname "_ss_i32")]] [[kernel]] decltype(binary_ss_ss<op, int, int>) binary_ss_ss<op, int, int>;

#define numeric_cmp(opname, op) \
template [[host_name(#opname "_vv_f32")]] [[kernel]] decltype(binary_ss_vv<op, float, bool>) binary_ss_vv<op, float, bool>;     \
template [[host_name(#opname "_vv_i32")]] [[kernel]] decltype(binary_ss_vv<op, int, bool>) binary_ss_vv<op, int, bool>;         \
template [[host_name(#opname "_sv_f32")]] [[kernel]] decltype(binary_ss_sv<op, float, bool>) binary_ss_sv<op, float, bool>;     \
template [[host_name(#opname "_sv_i32")]] [[kernel]] decltype(binary_ss_sv<op, int, bool>) binary_ss_sv<op, int, bool>;         \
template [[host_name(#opname "_vs_f32")]] [[kernel]] decltype(binary_ss_vs<op, float, bool>) binary_ss_vs<op, float, bool>;     \
template [[host_name(#opname "_vs_i32")]] [[kernel]] decltype(binary_ss_vs<op, int, bool>) binary_ss_vs<op, int, bool>;         \
template [[host_name(#opname "_ss_f32")]] [[kernel]] decltype(binary_ss_ss<op, float, bool>) binary_ss_ss<op, float, bool>;     \
template [[host_name(#opname "_ss_i32")]] [[kernel]] decltype(binary_ss_ss<op, int, bool>) binary_ss_ss<op, int, bool>;

#define cmp_all(opname, op) \
template [[host_name(#opname "_vv_f32")]] [[kernel]] decltype(binary_ss_vv<op, float, bool>) binary_ss_vv<op, float, bool>;     \
template [[host_name(#opname "_vv_i32")]] [[kernel]] decltype(binary_ss_vv<op, int, bool>) binary_ss_vv<op, int, bool>;         \
template [[host_name(#opname "_vv_b8")]] [[kernel]] decltype(binary_ss_vv<op, bool, bool>) binary_ss_vv<op, bool, bool>;        \
template [[host_name(#opname "_sv_f32")]] [[kernel]] decltype(binary_ss_sv<op, float, bool>) binary_ss_sv<op, float, bool>;     \
template [[host_name(#opname "_sv_i32")]] [[kernel]] decltype(binary_ss_sv<op, int, bool>) binary_ss_sv<op, int, bool>;         \
template [[host_name(#opname "_sv_b8")]] [[kernel]] decltype(binary_ss_sv<op, bool, bool>) binary_ss_sv<op, bool, bool>;        \
template [[host_name(#opname "_vs_f32")]] [[kernel]] decltype(binary_ss_vs<op, float, bool>) binary_ss_vs<op, float, bool>;     \
template [[host_name(#opname "_vs_i32")]] [[kernel]] decltype(binary_ss_vs<op, int, bool>) binary_ss_vs<op, int, bool>;         \
template [[host_name(#opname "_vs_b8")]] [[kernel]] decltype(binary_ss_vs<op, bool, bool>) binary_ss_vs<op, bool, bool>;        \
template [[host_name(#opname "_ss_f32")]] [[kernel]] decltype(binary_ss_ss<op, float, bool>) binary_ss_ss<op, float, bool>;     \
template [[host_name(#opname "_ss_i32")]] [[kernel]] decltype(binary_ss_ss<op, int, bool>) binary_ss_ss<op, int, bool>;         \
template [[host_name(#opname "_ss_b8")]] [[kernel]] decltype(binary_ss_ss<op, bool, bool>) binary_ss_ss<op, bool, bool>;

numeric_binary(add, Add)
numeric_binary(sub, Sub)
numeric_binary(mul, Mul)
numeric_binary(div, Div)
cmp_all(eq, Eq)
cmp_all(neq, Neq)
numeric_cmp(lt, Lt)
numeric_cmp(gt, Gt)
numeric_cmp(leq, Leq)
numeric_cmp(geq, Geq)
