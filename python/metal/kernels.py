from python.xavier import Array
from python.metal.context import MTLContext
import MetalPerformanceShaders as MPS
import Metal
import numpy as np


def _alloc(arr: Array, size: int, ctx: MTLContext):
    if arr.is_contiguous():
        buff = ctx.device.newBufferWithBytesNoCopy_length_options_deallocator_(
            arr, size, Metal.MTLResourceStorageModeShared, None
        )
    else:
        buff = ctx.device.newBufferWithLength_options_(size, Metal.MTLResourceStorageModeShared)
        tmp = Array.from_buffer(buff.contents().as_buffer(size)).interpret_(arr.dtype())
        arr.copy_to(tmp)
    return buff


def ss_op(name: str, input: list[Array], output: Array, ctx: MTLContext) -> Array:
    cmd_buff, encoder, kernel = ctx.get_resources(name)
    inbuffs = []
    numel = input[0].shape().numel()
    insize = input[0].nbytes()
    for i in range(len(input)):
        inbuff = _alloc(input[i], insize, ctx)
        inbuffs.append(inbuff)
    outbuff = ctx.device.newBufferWithBytesNoCopy_length_options_deallocator_(
        output, output.nbytes(), Metal.MTLResourceStorageModeShared, None
    )
    for i, inbuff in enumerate(inbuffs):
        encoder.setBuffer_offset_atIndex_(inbuff, 0, i)
    encoder.setBuffer_offset_atIndex_(outbuff, 0, len(inbuffs))
    grid_size = Metal.MTLSizeMake(numel, 1, 1)
    thread_group_size = Metal.MTLSizeMake(kernel.pipeline_state.maxTotalThreadsPerThreadgroup(), 1, 1)
    encoder.dispatchThreads_threadsPerThreadgroup_(grid_size, thread_group_size)
    encoder.endEncoding()
    cmd_buff.commit()
    cmd_buff.waitUntilCompleted()


def constant(arr: Array, c: float, ctx: MTLContext) -> Array:
    cmd_buff, encoder, kernel = ctx.get_resources(f"constant_c_{arr.dtype()}")
    np_c = np.array([c], dtype=np.float32)
    c_buff = ctx.device.newBufferWithBytesNoCopy_length_options_deallocator_(
        np_c, np_c.nbytes, Metal.MTLResourceStorageModeShared, None
    )
    out_buff = ctx.device.newBufferWithBytesNoCopy_length_options_deallocator_(
        arr, arr.nbytes(), Metal.MTLResourceStorageModeShared, None
    )
    encoder.setBuffer_offset_atIndex_(c_buff, 0, 0)
    encoder.setBuffer_offset_atIndex_(out_buff, 0, 1)
    grid_size = Metal.MTLSizeMake(arr.numel(), 1, 1)
    thread_group_size = Metal.MTLSizeMake(kernel.pipeline_state.maxTotalThreadsPerThreadgroup(), 1, 1)
    encoder.dispatchThreads_threadsPerThreadgroup_(grid_size, thread_group_size)
    encoder.endEncoding()
    cmd_buff.commit()
    cmd_buff.waitUntilCompleted()


def arange(arr: Array, start: int, step: int, ctx: MTLContext) -> Array:
    cmd_buff, encoder, kernel = ctx.get_resources(f"arange_{arr.dtype()}")
    np_start = np.array([start], dtype=np.int32)
    np_step = np.array([step], dtype=np.int32)
    start_buff = ctx.device.newBufferWithBytesNoCopy_length_options_deallocator_(
        np_start, np_start.nbytes, Metal.MTLResourceStorageModeShared, None
    )
    step_buff = ctx.device.newBufferWithBytesNoCopy_length_options_deallocator_(
        np_step, np_step.nbytes, Metal.MTLResourceStorageModeShared, None
    )
    out_buff = ctx.device.newBufferWithBytesNoCopy_length_options_deallocator_(
        arr, arr.nbytes(), Metal.MTLResourceStorageModeShared, None
    )
    encoder.setBuffer_offset_atIndex_(start_buff, 0, 0)
    encoder.setBuffer_offset_atIndex_(step_buff, 0, 1)
    encoder.setBuffer_offset_atIndex_(out_buff, 0, 2)
    grid_size = Metal.MTLSizeMake(arr.numel(), 1, 1)
    thread_group_size = Metal.MTLSizeMake(kernel.pipeline_state.maxTotalThreadsPerThreadgroup(), 1, 1)
    encoder.dispatchThreads_threadsPerThreadgroup_(grid_size, thread_group_size)
    encoder.endEncoding()
    cmd_buff.commit()
    cmd_buff.waitUntilCompleted()
