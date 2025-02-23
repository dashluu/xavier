from python import xavier as xv
from python.xavier import Array
import Metal
from python.base import Context


class MTLKernel:
    def __init__(self, pipeline_state, dtype: xv.Dtype):
        self._pipeline_state = pipeline_state
        self._dtype = dtype

    @property
    def pipeline_state(self):
        return self._pipeline_state

    @property
    def dtype(self):
        return self.dtype


class MTLFusedKernel(MTLKernel):
    def __init__(self, src, pipeline_state, dtype: xv.Dtype):
        super().__init__(pipeline_state, dtype)
        self._src = src

    @property
    def src(self):
        return self._src


class MTLContext(Context):
    _initializer_ops = ["constant_c", "arange"]
    _unary_ops = ["exp", "log", "neg", "recip"]
    _binary_ops = ["add", "sub", "mul", "div", "eq", "neq", "lt", "gt", "leq", "geq"]
    _util_ops = ["copy"]

    def _init_kernel(self, ops, sparse=True):
        for op in ops:
            for k in xv.num_dtypes:
                name = f"{op}_{k.name()}"
                f = self._lib.newFunctionWithName_(name)
                self._kernels[name] = MTLKernel(self._device.newComputePipelineStateWithFunction_error_(f, None)[0], k)
                if sparse:
                    name = f"sparse_{op}_{k.name()}"
                    f = self._lib.newFunctionWithName_(name)
                    self._kernels[name] = MTLKernel(
                        self._device.newComputePipelineStateWithFunction_error_(f, None)[0], k
                    )

    def __init__(self, metallib: str):
        self._device = Metal.MTLCreateSystemDefaultDevice()
        self._lib, _ = self._device.newLibraryWithURL_error_(metallib, None)
        self._cmd_queue = self._device.newCommandQueue()
        self._kernels: dict[str, MTLKernel] = {}
        self._init_kernel(MTLContext._initializer_ops, sparse=False)
        self._init_kernel(MTLContext._unary_ops)
        self._init_kernel(MTLContext._binary_ops)
        self._init_kernel(MTLContext._util_ops)

    @property
    def device(self):
        return self._device

    @property
    def lib(self):
        return self._lib

    @property
    def cmd_queue(self):
        return self._cmd_queue

    @property
    def kernels(self):
        return self._kernels

    def register_kernel(self, name: str, kernel: MTLKernel):
        if name in self._kernels:
            raise Exception("Function already exists as pipeline state object.")
        self._kernels[name] = kernel

    def get_resources(self, name: str):
        # Reuse command queue
        cmd_buff = self.cmd_queue.commandBuffer()
        encoder = cmd_buff.computeCommandEncoder()
        # Reuse pipeline state based on name
        kernel = self._kernels[name]
        encoder.setComputePipelineState_(kernel.pipeline_state)
        return cmd_buff, encoder, kernel
