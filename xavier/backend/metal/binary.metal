#include <metal_stdlib>

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

#define binary_all(opname, op) \
template [[host_name(#opname "_f32")]] [[kernel]] decltype(binary_ss<op, float, float>) binary_ss<op, float, float>; \
template [[host_name(#opname "_i32")]] [[kernel]] decltype(binary_ss<op, int, int>) binary_ss<op, int, int>;

#define cmp_all(opname, op) \
template [[host_name(#opname "_f32")]] [[kernel]] decltype(binary_ss<op, float, bool>) binary_ss<op, float, bool>; \
template [[host_name(#opname "_i32")]] [[kernel]] decltype(binary_ss<op, int, bool>) binary_ss<op, int, bool>;

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
