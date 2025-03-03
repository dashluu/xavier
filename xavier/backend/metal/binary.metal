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
[[kernel]] void binary_ss(
    device T *input1 [[buffer(0)]],
    device T *input2 [[buffer(1)]],
    device R *output [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    output[id] = Op()(input1[id], input2[id]);
}

template <class Op, class T, class R>
[[kernel]] void self_binary_ss(
    device T *input [[buffer(0)]],
    device R *output [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    output[id] = Op()(output[id], input[id]);
}

template <class Op, class T, class R>
[[kernel]] void sparse_binary_ss(
    constant const uint *ndim [[buffer(0)]],
    constant const uint *shape1 [[buffer(1)]],
    constant const uint *stride1 [[buffer(2)]],
    constant const uint *shape2 [[buffer(3)]],
    constant const uint *stride2 [[buffer(4)]],
    device T *input1 [[buffer(5)]],
    device T *input2 [[buffer(6)]],
    device R *output [[buffer(7)]],
    uint id [[thread_position_in_grid]])
{
    uint idx1 = access(id, ndim, shape1, stride1);
    uint idx2 = access(id, ndim, shape2, stride2);
    output[id] = Op()(input1[idx1], input2[idx2]);
}

template <class Op, class T, class R>
[[kernel]] void sparse_self_binary_ss(
    constant const uint *ndim [[buffer(0)]],
    constant const uint *shape1 [[buffer(1)]],
    constant const uint *stride1 [[buffer(2)]],
    constant const uint *shape2 [[buffer(3)]],
    constant const uint *stride2 [[buffer(4)]],
    device T *input [[buffer(5)]],
    device R *output [[buffer(6)]],
    uint id [[thread_position_in_grid]])
{
    uint idx1 = access(id, ndim, shape1, stride1);
    uint idx2 = access(id, ndim, shape2, stride2);
    output[idx2] = Op()(output[idx2], input[idx1]);
}

#define binary_all(opname, op) \
template [[host_name(#opname "_f32")]] [[kernel]] decltype(binary_ss<op, float, float>) binary_ss<op, float, float>;                        \
template [[host_name(#opname "_i32")]] [[kernel]] decltype(binary_ss<op, int, int>) binary_ss<op, int, int>;                                \
template [[host_name("sparse_" #opname "_f32")]] [[kernel]] decltype(sparse_binary_ss<op, float, float>) sparse_binary_ss<op, float, float>;\
template [[host_name("sparse_" #opname "_i32")]] [[kernel]] decltype(sparse_binary_ss<op, int, int>) sparse_binary_ss<op, int, int>;        \
template [[host_name("self_" #opname "_f32")]] [[kernel]] decltype(self_binary_ss<op, float, float>) self_binary_ss<op, float, float>;                        \
template [[host_name("self_" #opname "_i32")]] [[kernel]] decltype(self_binary_ss<op, int, int>) self_binary_ss<op, int, int>;                                \
template [[host_name("sparse_self_" #opname "_f32")]] [[kernel]] decltype(sparse_self_binary_ss<op, float, float>) sparse_self_binary_ss<op, float, float>;\
template [[host_name("sparse_self_" #opname "_i32")]] [[kernel]] decltype(sparse_self_binary_ss<op, int, int>) sparse_self_binary_ss<op, int, int>;

#define cmp_all(opname, op) \
template [[host_name(#opname "_f32")]] [[kernel]] decltype(binary_ss<op, float, bool>) binary_ss<op, float, bool>;                          \
template [[host_name(#opname "_i32")]] [[kernel]] decltype(binary_ss<op, int, bool>) binary_ss<op, int, bool>;                              \
template [[host_name("sparse_" #opname "_f32")]] [[kernel]] decltype(sparse_binary_ss<op, float, bool>) sparse_binary_ss<op, float, bool>;  \
template [[host_name("sparse_" #opname "_i32")]] [[kernel]] decltype(sparse_binary_ss<op, int, bool>) sparse_binary_ss<op, int, bool>;

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
