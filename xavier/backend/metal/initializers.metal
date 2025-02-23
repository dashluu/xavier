#include <metal_stdlib>

template <class T>
[[kernel]] void constant_c(
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
template [[host_name("constant_c_" #tyname)]] [[kernel]] decltype(constant_c<ty>) constant_c<ty>; \
template [[host_name("arange_" #tyname)]] [[kernel]] decltype(arange<ty>) arange<ty>;

initializer_all(f32, float)
initializer_all(i32, int)