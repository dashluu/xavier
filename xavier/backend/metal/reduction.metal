struct Sum
{
    template <class T>
    T operator()(T lhs, T rhs) { return lhs + rhs; }
};

struct AtomicSum
{
    template <class T, class R>
    void operator()(volatile device R *output, T val)
    {
        atomic_fetch_add_explicit(output, val, metal::memory_order_relaxed);
    }
};

template <class Op, class AtomicOp, class T, class R>
kernel void reduce_to_one(
    const device T *input [[buffer(0)]],
    device R *output [[buffer(1)]],
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
    R val = input[gid];
    for (uint s = lsize/simd_size; s > 1; s /= simd_size)
    {
        // Perform per-SIMD partial reduction.
        for (uint offset = simd_size/2; offset > 0; offset /= 2) {
            val = Op()(val, metal::simd_shuffle_down(val, offset));
        }
        // Write per-SIMD partial reduction value to threadgroup memory.
        if (simd_lane_id == 0) {
            ldata[simd_group_id] = val;
        }
        // Wait for all partial reductions to complete.
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        val = (lid < s) ? ldata[lid] : 0;
    }
    // Perform final per-SIMD partial reduction to calculate
    // the threadgroup partial reduction result.
    for (uint offset = simd_size/2; offset > 0; offset /= 2) {
        val = Op()(val, metal::simd_shuffle_down(val, offset));
    }
    // Atomically update the reduction result.
    if (lid == 0) {
        AtomicOp()(output, val);
    }
}

#define reduce_all(opname, op, atomic_op) \
template [[host_name(#opname "_f32")]] [[kernel]] decltype(reduce_to_one<op, atomic_op, float, float>) reduce_to_one<op, atomic_op, float, float>;

reduce_all(sum, Sum, AtomicSum)