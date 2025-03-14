#include <metal_stdlib>

template <class T>
[[kernel]] void full(
    device T *c [[buffer(0)]],
    device T *output [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    output[id] = *c;
}

template <class T>
[[kernel]] void arange(
    device int *start [[buffer(0)]],
    device int *step [[buffer(1)]],
    device T *output [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    output[id] = *start + static_cast<int>(id) * *step;
}

#define initializer_all(opname, op) \
template [[host_name(#opname "_f32")]] [[kernel]] decltype(op<float>) op<float>;    \
template [[host_name(#opname "_i32")]] [[kernel]] decltype(op<int>) op<int>;        \
template [[host_name(#opname "_b8")]] [[kernel]] decltype(op<bool>) op<bool>;

#define initializer_numeric_all(opname, op) \
template [[host_name(#opname "_f32")]] [[kernel]] decltype(op<float>) op<float>;    \
template [[host_name(#opname "_i32")]] [[kernel]] decltype(op<int>) op<int>;

initializer_all(full, full)
initializer_numeric_all(arange, arange)