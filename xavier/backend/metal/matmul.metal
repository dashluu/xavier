#include "metal.h"

template <class T>
void matmul2d_naive(
    constant const uint *shape1 [[buffer(0)]],
    constant const uint *shape2 [[buffer(1)]],
    device T *input1 [[buffer(2)]],
    device T *input2 [[buffer(3)]],
    device T *output [[buffer(4)]],
    uint2 id [[thread_position_in_grid]])
{
    auto row_dim = shape1[0];
    auto col_dim = shape2[1];
    auto inner_dim = shape1[1];
    if (id.x < col_dim && id.y < row_dim) {
        auto index = id.y * col_dim + id.x;
        T sum = 0;
        for (int i = 0; i < inner_dim; i++) {
            sum += input1[id.y * inner_dim + i] * input2[col_dim * i + id.x];
        }
        output[index] = sum;
    }
}