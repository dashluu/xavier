import time
from python.xavier import MTLGraph, Array, MTLContext
import numpy as np

ctx = MTLContext("./xavier/build/backend/metal/kernels.metallib")
x1 = Array.full([1, 3, 2], 7.0)
x2 = Array.arange([1, 3, 2], 2, 2)
x3 = x1.sq()
g = MTLGraph(x3, ctx)
start = time.time()
g.compile()
g.forward()
end = time.time()
print(end - start)
print()
print(x1)
print()
# print(x2)
print()
print(x3)
