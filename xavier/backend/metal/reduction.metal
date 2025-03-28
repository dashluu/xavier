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

struct AtomicMax
{
    template <class T, class R>
    void operator()(volatile device metal::_atomic<R> *output, T val)
    {
        metal::atomic_fetch_max_explicit(output, val, metal::memory_order_relaxed);
    }
};

struct AtomicMin
{
    template <class T, class R>
    void operator()(volatile device metal::_atomic<R> *output, T val)
    {
        metal::atomic_fetch_min_explicit(output, val, metal::memory_order_relaxed);
    }
};

template <class Op, class AtomicOp, class T, class R>
kernel void reduce_to_one(
    constant const uint *offset [[buffer(0)]],
    const device T *input [[buffer(1)]],
    device metal::_atomic<R> *output [[buffer(2)]],
    threadgroup T *ldata [[threadgroup(0)]],
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
kernel void strided_reduce_to_one(
    constant const uint *ndim [[buffer(0)]],
    constant const uint *offset [[buffer(1)]],
    constant const uint *shape [[buffer(2)]],
    constant const int *stride [[buffer(3)]],
    const device T *input [[buffer(4)]],
    device metal::_atomic<R> *output [[buffer(5)]],
    threadgroup T *ldata [[threadgroup(0)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_size [[threads_per_simdgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    uint idx = strided_idx(gid, ndim, shape, stride);
    R val = input[offset[0] + idx];
    for (uint s = lsize/simd_size; s > 1; s /= simd_size)
    {
        for (uint lanes = simd_size/2; lanes > 0; lanes /= 2) {
            if (gid >= lanes) {
                uint shuffle_dist = idx - strided_idx(gid - lanes, ndim, shape, stride);
                val = Op()(val, metal::simd_shuffle_down(val, shuffle_dist));
            }
        }
        if (simd_lane_id == 0) {
            ldata[simd_group_id] = val;
        }
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        val = (lid < s) ? ldata[lid] : 0;
    }
    for (uint lanes = simd_size/2; lanes > 0; lanes /= 2) {
        if (gid >= lanes) {
            uint shuffle_dist = idx - strided_idx(gid - lanes, ndim, shape, stride);
            val = Op()(val, metal::simd_shuffle_down(val, shuffle_dist));
        }
    }
    if (lid == 0) {
        AtomicOp()(output + offset[1], val);
    }
}

#define reduce_all(opname, op, atomic_op) \
template [[host_name(#opname "_f32")]] [[kernel]] decltype(reduce_to_one<op, atomic_op, float, float>) reduce_to_one<op, atomic_op, float, float>;                              \
template [[host_name(#opname "_i32")]] [[kernel]] decltype(reduce_to_one<op, atomic_op, int, int>) reduce_to_one<op, atomic_op, int, int>;                                      \
template [[host_name("strided_" #opname "_f32")]] [[kernel]] decltype(strided_reduce_to_one<op, atomic_op, float, float>) strided_reduce_to_one<op, atomic_op, float, float>;   \
template [[host_name("strided_" #opname "_i32")]] [[kernel]] decltype(strided_reduce_to_one<op, atomic_op, int, int>) strided_reduce_to_one<op, atomic_op, int, int>;

reduce_all(sum, Sum, AtomicSum)
// reduce_all(max, Max, AtomicMax)
// reduce_all(min, Min, AtomicMin)