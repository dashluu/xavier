#include "utils.h"

struct Sum
{
    template <class T>
    T operator()(T lhs, T rhs) { return lhs + rhs; }
};

struct Max
{
    template <class T>
    T operator()(T lhs, T rhs) { return lhs <= rhs ? rhs : lhs; }
};

struct Min
{
    template <class T>
    T operator()(T lhs, T rhs) { return lhs <= rhs ? lhs : rhs; }
};

struct AtomicSum
{
    template <class T, class R>
    void operator()(volatile device metal::_atomic<R> *output, T val)
    {
        // memory_order_relaxed guarantees atomicity without ordering or proper synchronization
        // since we're doing addition, this is somewhat similar to a counter
        // atomic_fetch_add_explicit runs output += val but atomically
        metal::atomic_fetch_add_explicit(output, val, metal::memory_order_relaxed);
    }
};

struct AtomicMaxInt {
    template <class T, class R>
    void operator()(volatile device metal::_atomic<R> *output, T val)
    {
        metal::atomic_fetch_max_explicit(output, val, metal::memory_order_relaxed);
    }
};

struct AtomicMaxFloat
{
    template <class T, class R>
    void operator()(volatile device metal::_atomic<R> *output, T val)
    {
        // Hackery way using atomic_fetch_max_explicit, atomic_fetch_min_explicit for int
        // TODO: handle nan and inf case?
        if (!metal::signbit(val)) {
            metal::atomic_fetch_max_explicit(reinterpret_cast<volatile device metal::_atomic<int>*>(output), as_type<int>(val), metal::memory_order_relaxed);
        } else {
            metal::atomic_fetch_min_explicit(reinterpret_cast<volatile device metal::_atomic<uint>*>(output), as_type<uint>(val), metal::memory_order_relaxed);
        }
    }
};

template <class Op, class AtomicOp, class T, class R>
kernel void reduce_all_vv(
    constant const uint *offset [[buffer(0)]],
    const device T *input [[buffer(1)]],
    device metal::_atomic<R> *output [[buffer(2)]],
    threadgroup R *ldata [[threadgroup(0)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_size [[threads_per_simdgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    // Perform the first level of reduction.
    // Read from device memory, write to threadgroup memory.
    // val is stored in thread's register
    R val = input[offset[0] + gid];
    for (uint s = lsize/simd_size; s > 1; s /= simd_size)
    {
        // Perform per-SIMD partial reduction -> shuffling within SIMD group.
        // Each thread gets the value from another thread offset lanes above it.
        // Threads with index < offset lanes keep their original values.
        /*
        simd_shuffle_down with simd_size = 8:
        Initial values in registers:
        Thread ID:    0  1  2  3  4  5  6  7
        Values:       a  b  c  d  e  f  g  h

        After simd_shuffle_down(val, 4):
        Thread ID:    0  1  2  3  4  5  6  7
        Values:       e  f  g  h  e  f  g  h

        After simd_shuffle_down(val, 2):
        Thread ID:    0  1  2  3  4  5  6  7
        Values:       c  d  g  h  e  f  g  h

        After simd_shuffle_down(val, 1):
        Thread ID:    0  1  2  3  4  5  6  7
        Values:       b  d  g  h  e  f  g  h
        */
        for (uint lanes = simd_size/2; lanes > 0; lanes /= 2) {
            val = Op()(val, metal::simd_shuffle_down(val, lanes));
        }
        // Write per-SIMD partial reduction value to threadgroup memory.
        if (simd_lane_id == 0) {
            ldata[simd_group_id] = val;
        }
        // Wait for all partial reductions to complete.
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        val = (lid < s) ? ldata[lid] : 0;
    }
    // Perform final per-SIMD partial reduction to calculate the threadgroup partial reduction result.
    for (uint lanes = simd_size/2; lanes > 0; lanes /= 2) {
        val = Op()(val, metal::simd_shuffle_down(val, lanes));
    }
    // Atomically update the reduction result.
    if (lid == 0) {
        AtomicOp()(output + offset[1], val);
    }
}

template <class Op, class AtomicOp, class T, class R>
kernel void reduce_all_vs(
    constant const uint *ndim [[buffer(0)]],
    constant const uint *offset [[buffer(1)]],
    constant const uint *shape [[buffer(2)]],
    constant const int *stride [[buffer(3)]],
    const device T *input [[buffer(4)]],
    device metal::_atomic<R> *output [[buffer(5)]],
    threadgroup R *ldata [[threadgroup(0)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_size [[threads_per_simdgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    // The algorithm is same as before with the exception that
    // elements are accessed non-contiguously
    uint idx = strided_idx(gid, ndim, shape, stride);
    R val = input[offset[0] + idx];
    for (uint s = lsize/simd_size; s > 1; s /= simd_size)
    {
        for (uint lanes = simd_size/2; lanes > 0; lanes /= 2) {
            val = Op()(val, metal::simd_shuffle_down(val, lanes));
        }
        if (simd_lane_id == 0) {
            ldata[simd_group_id] = val;
        }
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        val = (lid < s) ? ldata[lid] : 0;
    }
    for (uint lanes = simd_size/2; lanes > 0; lanes /= 2) {
        val = Op()(val, metal::simd_shuffle_down(val, lanes));
    }
    if (lid == 0) {
        AtomicOp()(output + offset[1], val);
    }
}

template <class Op, class AtomicOp, class T, class R>
kernel void reduce_col_vv(
    constant const uint *offset [[buffer(0)]],
    constant const uint *shape [[buffer(1)]],
    const device T *input [[buffer(2)]],
    device metal::_atomic<R> *output [[buffer(3)]],
    threadgroup R *ldata [[threadgroup(0)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint2 lsize [[threads_per_threadgroup]],
    uint simd_size [[threads_per_simdgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    const uint grow = gid.y;
    const uint gcol = gid.x;
    const uint lrow = lid.y;
    const uint lcol = lid.x;
    const uint lheight = lsize.y;
    const uint lwidth = lsize.x;
    const uint M = shape[0];
    const uint N = shape[1];
    R val = input[offset[0] + grow * N + gcol];
    for (uint s = lwidth/simd_size; s > 1; s /= simd_size)
    {
        for (uint lanes = simd_size/2; lanes > 0; lanes /= 2) {
            if (lanes < N) {
                val = Op()(val, metal::simd_shuffle_down(val, lanes));
            }
        }
        if (simd_lane_id == 0) {
            ldata[lrow * lwidth + simd_group_id] = val;
        }
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        val = (lcol < s) ? ldata[lrow * lwidth + lcol] : 0;
    }
    for (uint lanes = simd_size/2; lanes > 0; lanes /= 2) {
        if (lanes < N) {
            val = Op()(val, metal::simd_shuffle_down(val, lanes));
        }
    }
    if (lcol == 0) {
        AtomicOp()(output + offset[1] + grow, val);
    }
}

#define reduce(opname, op, atomic_op_float, atomic_op_int) \
template [[host_name(#opname "_all_vv_f32")]] [[kernel]] decltype(reduce_all_vv<op, atomic_op_float, float, float>) reduce_all_vv<op, atomic_op_float, float, float>;   \
template [[host_name(#opname "_all_vv_i32")]] [[kernel]] decltype(reduce_all_vv<op, atomic_op_int, int, int>) reduce_all_vv<op, atomic_op_int, int, int>;               \
template [[host_name(#opname "_all_vs_f32")]] [[kernel]] decltype(reduce_all_vs<op, atomic_op_float, float, float>) reduce_all_vs<op, atomic_op_float, float, float>;   \
template [[host_name(#opname "_all_vs_i32")]] [[kernel]] decltype(reduce_all_vs<op, atomic_op_int, int, int>) reduce_all_vs<op, atomic_op_int, int, int>;               \
template [[host_name(#opname "_col_vv_f32")]] [[kernel]] decltype(reduce_col_vv<op, atomic_op_float, float, float>) reduce_col_vv<op, atomic_op_float, float, float>;   \
template [[host_name(#opname "_col_vv_i32")]] [[kernel]] decltype(reduce_col_vv<op, atomic_op_int, int, int>) reduce_col_vv<op, atomic_op_int, int, int>;

reduce(sum, Sum, AtomicSum, AtomicSum)
reduce(max, Max, AtomicMaxFloat, AtomicMaxInt)