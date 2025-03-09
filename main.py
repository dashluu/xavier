import time
from python.xavier import MTLGraph, Array, MTLContext, b8, i32, f32
import numpy as np

ctx = MTLContext("./xavier/build/backend/metal/kernels.metallib")
x1 = Array.full([2, 1, 4], 7.0, dtype=f32)
x2 = Array.arange([2, 1, 4], 2, 2, dtype=f32)
x3 = x1 + x2
g = MTLGraph(x3, ctx)
g.compile()
print(g)
g.forward()
print()
print(x3)
# ctx = MTLContext("./xavier/build/backend/metal/kernels.metallib")
# x1 = Array.arange([2, 1, 4], 2, 2)
# g1 = MTLGraph(x1, ctx)
# g1.compile()
# g1.forward()
# print(x1)
# x2 = Array.zeros_like(x1, dtype=b8)
# g2 = MTLGraph(x2, ctx)
# g2.compile()
# g2.forward()
# print(x2)
