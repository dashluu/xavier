import time
from python.xavier import MTLGraph, Array, MTLContext, b8, i32, f32
import numpy as np

ctx = MTLContext("./xavier/build/backend/metal/kernels.metallib")
x1 = Array.full([2, 1, 4], 7.0, dtype=f32)
# x2 = Array.arange([2, 1, 4], 2, 2, dtype=i32)
x2 = Array.full([2, 1, 4], 9.0, dtype=f32)
x3 = 2 * x1
g = MTLGraph(x3, ctx)
g.compile()
print(g)
g.forward()
print()
print(x3)
