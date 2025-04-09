import time
from python.xavier import MTLGraph, Array, MTLContext, b8, i32, f32
import numpy as np
import torch
import time

ctx = MTLContext("./xavier/build/backend/metal/kernels.metallib")
x = torch.randn((384000, 2000), dtype=torch.float32)
arr1 = Array.from_numpy(x.numpy())
arr2 = arr1.sum([1])
arr3 = arr2.sum()
g = MTLGraph(arr3, ctx)
g.compile()
start = time.time()
g.forward()
end = time.time()
print(end - start)
start = time.time()
expected = x.sum(dim=-1)
end = time.time()
print(end - start)
# print(arr1.numpy())
# print(arr2.numpy().flatten())
# print(expected)
