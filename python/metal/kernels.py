from python.xavier import Array
from python.metal.context import MTLContext
import MetalPerformanceShaders as MPS
import Metal
import numpy as np


def ss_op(name: str, input: list[Array], output: Array, ctx: MTLContext) -> Array:
    cmd_buff, encoder, kernel = ctx.get_resources(f"{name}_{output.dtype()}")
    buff_idx = 0
    for i in range(len(input)):
        inbuff = ctx.device.newBufferWithBytesNoCopy_length_options_deallocator_(
            input[i], input[i].nbytes(), Metal.MTLResourceStorageModeShared, None
        )
        encoder.setBuffer_offset_atIndex_(inbuff, 0, buff_idx)
        buff_idx += 1
    outbuff = ctx.device.newBufferWithBytesNoCopy_length_options_deallocator_(
        output, output.nbytes(), Metal.MTLResourceStorageModeShared, None
    )
    encoder.setBuffer_offset_atIndex_(outbuff, 0, buff_idx)
    grid_size = Metal.MTLSizeMake(output.numel(), 1, 1)
    thread_group_size = Metal.MTLSizeMake(kernel.pipeline_state.maxTotalThreadsPerThreadgroup(), 1, 1)
    encoder.dispatchThreads_threadsPerThreadgroup_(grid_size, thread_group_size)
    encoder.endEncoding()
    cmd_buff.commit()
    cmd_buff.waitUntilCompleted()


def sparse_ss_op(name: str, input: list[Array], output: Array, ctx: MTLContext) -> Array:
    cmd_buff, encoder, kernel = ctx.get_resources(f"sparse_{name}_{output.dtype()}")
    buff_idx = 0
    # Input # dimensions
    np_ndim = np.array([output.ndim()], dtype=np.uint32)
    ndim_buff = ctx.device.newBufferWithBytesNoCopy_length_options_deallocator_(
        np_ndim, np_ndim.nbytes, Metal.MTLResourceStorageModeShared, None
    )
    encoder.setBuffer_offset_atIndex_(ndim_buff, 0, buff_idx)
    buff_idx += 1
    # Input's shape, stride
    for i in range(len(input)):
        shape = input[i].shape()
        np_shape = np.array(shape.view(), dtype=np.uint32)
        shape_buff = ctx.device.newBufferWithBytesNoCopy_length_options_deallocator_(
            np_shape, np_shape.nbytes, Metal.MTLResourceStorageModeShared, None
        )
        encoder.setBuffer_offset_atIndex_(shape_buff, 0, buff_idx)
        buff_idx += 1
        np_stride = np.array(shape.stride(), dtype=np.uint32)
        stride_buff = ctx.device.newBufferWithBytesNoCopy_length_options_deallocator_(
            np_stride, np_stride.nbytes, Metal.MTLResourceStorageModeShared, None
        )
        encoder.setBuffer_offset_atIndex_(stride_buff, 0, buff_idx)
        buff_idx += 1
    # Set up input buffer
    for i in range(len(input)):
        input_buff = ctx.device.newBufferWithBytesNoCopy_length_options_deallocator_(
            input[i], input[i].nbytes(), Metal.MTLResourceStorageModeShared, None
        )
        encoder.setBuffer_offset_atIndex_(input_buff, 0, buff_idx)
        buff_idx += 1
    # Set up output buffer
    output_buff = ctx.device.newBufferWithBytesNoCopy_length_options_deallocator_(
        output, output.nbytes(), Metal.MTLResourceStorageModeShared, None
    )
    encoder.setBuffer_offset_atIndex_(output_buff, 0, buff_idx)
    buff_idx += 1
    grid_size = Metal.MTLSizeMake(shape.numel(), 1, 1)
    thread_group_size = Metal.MTLSizeMake(kernel.pipeline_state.maxTotalThreadsPerThreadgroup(), 1, 1)
    encoder.dispatchThreads_threadsPerThreadgroup_(grid_size, thread_group_size)
    encoder.endEncoding()
    cmd_buff.commit()
    cmd_buff.waitUntilCompleted()


def copy(input: Array, output: Array, ctx: MTLContext) -> Array:
    cmd_buff, encoder, kernel = ctx.get_resources(f"copy_{input.dtype()}")
    buff_idx = 0
    inbuff = ctx.device.newBufferWithBytesNoCopy_length_options_deallocator_(
        input, input.nbytes(), Metal.MTLResourceStorageModeShared, None
    )
    encoder.setBuffer_offset_atIndex_(inbuff, 0, buff_idx)
    buff_idx += 1
    outbuff = ctx.device.newBufferWithBytesNoCopy_length_options_deallocator_(
        output, output.nbytes(), Metal.MTLResourceStorageModeShared, None
    )
    encoder.setBuffer_offset_atIndex_(outbuff, 0, buff_idx)
    grid_size = Metal.MTLSizeMake(output.numel(), 1, 1)
    thread_group_size = Metal.MTLSizeMake(kernel.pipeline_state.maxTotalThreadsPerThreadgroup(), 1, 1)
    encoder.dispatchThreads_threadsPerThreadgroup_(grid_size, thread_group_size)
    encoder.endEncoding()
    cmd_buff.commit()
    cmd_buff.waitUntilCompleted()


def sparse_copy(input: Array, output: Array, ctx: MTLContext) -> Array:
    cmd_buff, encoder, kernel = ctx.get_resources(f"sparse_copy_{input.dtype()}")
    buff_idx = 0
    shape = input.shape()
    # Input # dimensions
    np_ndim = np.array([shape.ndim()], dtype=np.uint32)
    ndim_buff = ctx.device.newBufferWithBytesNoCopy_length_options_deallocator_(
        np_ndim, np_ndim.nbytes, Metal.MTLResourceStorageModeShared, None
    )
    encoder.setBuffer_offset_atIndex_(ndim_buff, 0, buff_idx)
    buff_idx += 1
    # Input's shape, stride
    np_shape = np.array(shape.view(), dtype=np.uint32)
    shape_buff = ctx.device.newBufferWithBytesNoCopy_length_options_deallocator_(
        np_shape, np_shape.nbytes, Metal.MTLResourceStorageModeShared, None
    )
    encoder.setBuffer_offset_atIndex_(shape_buff, 0, buff_idx)
    buff_idx += 1
    np_stride = np.array(shape.stride(), dtype=np.uint32)
    stride_buff = ctx.device.newBufferWithBytesNoCopy_length_options_deallocator_(
        np_stride, np_stride.nbytes, Metal.MTLResourceStorageModeShared, None
    )
    encoder.setBuffer_offset_atIndex_(stride_buff, 0, buff_idx)
    buff_idx += 1
    # Set up input buffer
    input_buff = ctx.device.newBufferWithBytesNoCopy_length_options_deallocator_(
        input, input.nbytes(), Metal.MTLResourceStorageModeShared, None
    )
    encoder.setBuffer_offset_atIndex_(input_buff, 0, buff_idx)
    buff_idx += 1
    # Set up output buffer
    output_buff = ctx.device.newBufferWithBytesNoCopy_length_options_deallocator_(
        output, output.nbytes(), Metal.MTLResourceStorageModeShared, None
    )
    encoder.setBuffer_offset_atIndex_(output_buff, 0, buff_idx)
    buff_idx += 1
    grid_size = Metal.MTLSizeMake(shape.numel(), 1, 1)
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
