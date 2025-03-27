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
kernel void binary_ss(
    constant const uint *offset [[buffer(0)]],
    device T *lhs [[buffer(1)]],
    device T *rhs [[buffer(2)]],
    device R *output [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    output[offset[2] + id] = Op()(lhs[offset[0] + id], rhs[offset[1] + id]);
}

template <class Op, class T, class R>
kernel void strided_binary_ss(
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
    uint lhs_idx = access(id, ndim, shape, lhs_stride);
    uint rhs_idx = access(id, ndim, shape, rhs_stride);
    output[offset[2] + id] = Op()(lhs[offset[0] + lhs_idx], rhs[offset[1] + rhs_idx]);
}

#define binary_all(opname, op) \
template [[host_name(#opname "_f32")]] [[kernel]] decltype(binary_ss<op, float, float>) binary_ss<op, float, float>;                            \
template [[host_name(#opname "_i32")]] [[kernel]] decltype(binary_ss<op, int, int>) binary_ss<op, int, int>;                                    \
template [[host_name("strided_" #opname "_f32")]] [[kernel]] decltype(strided_binary_ss<op, float, float>) strided_binary_ss<op, float, float>; \
template [[host_name("strided_" #opname "_i32")]] [[kernel]] decltype(strided_binary_ss<op, int, int>) strided_binary_ss<op, int, int>;

#define cmp_all(opname, op) \
template [[host_name(#opname "_f32")]] [[kernel]] decltype(binary_ss<op, float, bool>) binary_ss<op, float, bool>;                              \
template [[host_name(#opname "_i32")]] [[kernel]] decltype(binary_ss<op, int, bool>) binary_ss<op, int, bool>;                                  \
template [[host_name("strided_" #opname "_f32")]] [[kernel]] decltype(strided_binary_ss<op, float, bool>) strided_binary_ss<op, float, bool>;   \
template [[host_name("strided_" #opname "_i32")]] [[kernel]] decltype(strided_binary_ss<op, int, bool>) strided_binary_ss<op, int, bool>;

binary_all(add, Add)
binary_all(sub, Sub)
binary_all(mul, Mul)
binary_all(div, Div)
cmp_all(eq, Eq)
cmp_all(neq, Neq)
cmp_all(lt, Lt)
cmp_all(gt, Gt)
cmp_all(leq, Leq)
cmp_all(geq, Geq)
