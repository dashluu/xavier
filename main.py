import time
from python.xavier import MTLGraph, Array, MTLContext, b8, i32, f32
import numpy as np
import torch

ctx = MTLContext("./xavier/build/backend/metal/kernels.metallib")
x = torch.arange(0, 120, dtype=torch.float32).reshape(2, 3, 4, 5)
arr1 = Array.from_numpy(x.numpy())
arr2 = arr1[1::, ::2, ::3]
arr3 = arr2.sum()
g = MTLGraph(arr3, ctx)
g.compile()
g.forward()
t1 = x.requires_grad_(True)
t2 = t1[1::, ::2, ::3]
t3 = t2.sum()
print(arr1)
print(arr2)
print(t2)
print(arr3)
print(t3)
