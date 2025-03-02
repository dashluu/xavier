#include <metal_stdlib>

template <class T>
[[kernel]] void full(
    device float *c [[buffer(0)]],
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
    output[id] = *start + id * *step;
}

#define initializer_all(tyname, ty) \
template [[host_name("full_" #tyname)]] [[kernel]] decltype(full<ty>) full<ty>; \
template [[host_name("arange_" #tyname)]] [[kernel]] decltype(arange<ty>) arange<ty>;

initializer_all(f32, float)
initializer_all(i32, int)