import time
from python.xavier import MTLGraph, Array, MTLContext, b8, i32, f32
import numpy as np
import torch

ctx = MTLContext("./xavier/build/backend/metal/kernels.metallib")
x = torch.randn([5, 42], dtype=torch.float32)
arr1 = Array.from_numpy(x.numpy())
arr2 = arr1.sum([1])
arr3 = arr2.sum()
g = MTLGraph(arr3, ctx)
g.compile()
g.forward()
expected = x.sum(dim=-1)
# print(arr1)
print(arr2)
print(expected)
