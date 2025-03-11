#include "metal.h"

template <class T, class R>
void matmul2d(
    constant const uint *offset [[buffer(0)]],
    constant const uint *lhs_shape [[buffer(1)]],
    constant const uint *rhs_shape [[buffer(2)]],
    device T *lhs [[buffer(3)]],
    device T *rhs [[buffer(4)]],
    device R *output [[buffer(5)]],
    uint2 id [[thread_position_in_grid]])
{
    auto row_dim = lhs_shape[0];
    auto col_dim = rhs_shape[1];
    auto inner_dim = lhs_shape[1];
    if (id.x < col_dim && id.y < row_dim) {
        auto index = offset[2] + id.y * col_dim + id.x;
        R sum = 0;
        for (int i = 0; i < inner_dim; i++) {
            sum += lhs[offset[0] + id.y * inner_dim + i] * rhs[offset[1] + col_dim * i + id.x];
        }
        output[index] = sum;
    }
}

template <class T, class R>
void matmul3d(
    constant const uint *offset [[buffer(0)]],
    constant const uint *lhs_shape [[buffer(1)]],
    constant const uint *rhs_shape [[buffer(2)]],
    device T *lhs [[buffer(3)]],
    device T *rhs [[buffer(4)]],
    device R *output [[buffer(5)]],
    uint3 id [[thread_position_in_grid]])
{
    auto batch_dim = lhs_shape[0];
    auto row_dim = lhs_shape[1];
    auto col_dim = rhs_shape[2];
    auto inner_dim = lhs_shape[2];
    if (id.x < col_dim && id.y < row_dim && id.z < batch_dim) {
        auto index = offset[2] + id.z * batch_dim + id.y * col_dim + id.x;
        R sum = 0;
        for (int i = 0; i < inner_dim; i++) {
            sum += lhs[offset[0] + id.z * batch_dim + id.y * inner_dim + i] * rhs[offset[1] + id.z * batch_dim + col_dim * i + id.x];
        }
        output[index] = sum;
    }
}

template [[host_name("matmul2d_f32")]] [[kernel]] decltype(matmul2d<op, float, float>) matmul2d<op, float, float>;
template [[host_name("matmul2d_i32")]] [[kernel]] decltype(matmul2d<op, int, int>) matmul2d<op, int, int>;