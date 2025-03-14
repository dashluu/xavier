import time
from python.xavier import MTLGraph, Array, MTLContext, b8, i32, f32
import numpy as np

ctx = MTLContext("./xavier/build/backend/metal/kernels.metallib")
x1 = Array.full([8, 8], 7.0, dtype=f32)
x2 = Array.arange([8, 8], 0, 1, dtype=f32)
x3 = x1 @ x2
g = MTLGraph(x3, ctx)
g.compile()
print(g)
g.forward()
print()
print(x1)
print()
print(x2)
print()
print(x3)
