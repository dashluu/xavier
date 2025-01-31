import sys
from python.config import config
from python.metal.context import MTLContext
from python.metal.compile import MTLCompiler


def _init():
    if sys.platform != "darwin":
        raise Exception("Metal is only compatible with the current device.")
    else:
        config.ctx = MTLContext("./xavier/build/backend/metal/kernels.metallib")
        config.compiler = MTLCompiler(config.ctx)


_init()

__all__ = ["xavier", "config"]
