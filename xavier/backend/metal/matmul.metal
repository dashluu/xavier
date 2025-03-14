#include "utils.h"

template <class T, class R>
[[kernel]] void matmul2d(
    constant const uint *offset [[buffer(0)]],
    constant const uint *lhs_shape [[buffer(1)]],
    constant const uint *rhs_shape [[buffer(2)]],
    device T *lhs [[buffer(3)]],
    device T *rhs [[buffer(4)]],
    device R *output [[buffer(5)]],
    uint2 id [[thread_position_in_grid]])
{
    const uint row = id.y;
    const uint col = id.x;
    const uint M = lhs_shape[0];
    const uint N = rhs_shape[1];
    const uint K = lhs_shape[1];
    if (row < M && col < N) {
        // Calculate output index
        const uint out_idx = row * N + col;
        // Matrix multiplication
        R sum = 0;
        for (uint k = 0; k < K; k++) {
            const uint lhs_idx = offset[0] + row * K + k;
            const uint rhs_idx = offset[1] + k * N + col;
            sum += lhs[lhs_idx] * rhs[rhs_idx];
        }
        output[out_idx] = sum;
    }
}

template <class T, class R>
[[kernel]] void matmul3d(
    constant const uint *offset [[buffer(0)]],
    constant const uint *lhs_shape [[buffer(1)]],
    constant const uint *rhs_shape [[buffer(2)]],
    device T *lhs [[buffer(3)]],
    device T *rhs [[buffer(4)]],
    device R *output [[buffer(5)]],
    uint3 id [[thread_position_in_grid]])
{
    const uint batch = id.z;
    const uint row = id.y;
    const uint col = id.x;
    // Get dimensions
    const uint B = lhs_shape[0];  // Batch size
    const uint M = lhs_shape[1];  // Rows in each matrix
    const uint N = rhs_shape[2];  // Cols in each matrix
    const uint K = lhs_shape[2];  // Inner dimension
    if (col < N && row < M && batch < B) {
        // Calculate output index
        // [batch, row, col] -> batch * (M * N) + row * N + col
        const uint out_idx = batch * M * N + row * N + col;
        R sum = 0;
        for (int i = 0; i < K; i++) {
            // [batch, row, k] -> batch * (M * K) + row * K + k
            const uint lhs_idx = offset[0] + batch * M * K + row * K + i;
            // [batch, k, col] -> batch * (K * N) + k * N + col
            const uint rhs_idx = offset[1] + batch * K * N + N * i + col;
            sum += lhs[lhs_idx] * rhs[rhs_idx];
        }
        output[out_idx] = sum;
    }
}

template [[host_name("matmul2d_f32")]] [[kernel]] decltype(matmul2d<float, float>) matmul2d<float, float>;
template [[host_name("matmul2d_i32")]] [[kernel]] decltype(matmul2d<int, int>) matmul2d<int, int>;
template [[host_name("matmul3d_f32")]] [[kernel]] decltype(matmul3d<float, float>) matmul3d<float, float>;
template [[host_name("matmul3d_i32")]] [[kernel]] decltype(matmul3d<int, int>) matmul3d<int, int>;