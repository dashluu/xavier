#include "utils.h"

template <class T, class R>
kernel void copy(
    constant const uint *offset [[buffer(0)]],
    device T *src [[buffer(1)]],
    device R *dst [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    dst[offset[1] + id] = src[offset[0] + id];
}

template <class T, class R>
kernel void strided_copy(
    constant const uint *ndim [[buffer(0)]],
    constant const uint *offset [[buffer(1)]],
    constant const uint *shape [[buffer(2)]],
    constant const int *stride [[buffer(3)]],
    device T *src [[buffer(4)]],
    device R *dst [[buffer(5)]],
    uint id [[thread_position_in_grid]])
{
    uint idx = access(id, ndim, shape, stride);
    dst[offset[1] + id] = src[offset[0] + idx];
}

#define copy_all() \
template [[host_name("copy_f32")]] [[kernel]] decltype(copy<float, float>) copy<float, float>;                          \
template [[host_name("copy_i32")]] [[kernel]] decltype(copy<int, int>) copy<int, int>;                                  \
template [[host_name("copy_b8")]] [[kernel]] decltype(copy<bool, bool>) copy<bool, bool>;                               \
template [[host_name("strided_copy_f32")]] [[kernel]] decltype(strided_copy<float, float>) strided_copy<float, float>;  \
template [[host_name("strided_copy_i32")]] [[kernel]] decltype(strided_copy<int, int>) strided_copy<int, int>;          \
template [[host_name("strided_copy_b8")]] [[kernel]] decltype(strided_copy<bool, bool>) strided_copy<bool, bool>;

copy_all()