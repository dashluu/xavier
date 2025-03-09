import time
from python.xavier import MTLGraph, Array, MTLContext
import numpy as np

ctx = MTLContext("./xavier/build/backend/metal/kernels.metallib")
x1 = Array.full([2, 1, 4], 7.0)
x2 = Array.arange([2, 1, 4], 2, 2)
x1 += x2
g = MTLGraph(x1, ctx)
g.compile()
print(g)
g.forward()
print()
print(x2)
print()
print(x1)
# print()
# print(x3)
