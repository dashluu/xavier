#include "utils.h"

template <class T>
[[kernel]] void copy(
    device T *input [[buffer(0)]],
    device T *output [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    output[id] = input[id];
}

template <class T>
[[kernel]] void sparse_copy(
    constant const uint *ndim [[buffer(0)]],
    constant const uint *shape [[buffer(1)]],
    constant const uint *stride [[buffer(2)]],
    device T *input [[buffer(3)]],
    device T *output [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    uint idx = access(id, ndim, shape, stride);
    output[id] = input[idx];
}

#define copy_all() \
template [[host_name("copy_f32")]] [[kernel]] decltype(copy<float>) copy<float>;                        \
template [[host_name("copy_i32")]] [[kernel]] decltype(copy<int>) copy<int>;                            \
template [[host_name("sparse_copy_f32")]] [[kernel]] decltype(sparse_copy<float>) sparse_copy<float>;   \
template [[host_name("sparse_copy_i32")]] [[kernel]] decltype(sparse_copy<int>) sparse_copy<int>;

copy_all()