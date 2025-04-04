import time
from python.xavier import MTLGraph, Array, MTLContext, b8, i32, f32
import numpy as np
import torch

ctx = MTLContext("./xavier/build/backend/metal/kernels.metallib")
x = torch.arange(0, 60, dtype=torch.float32).reshape(3, 4, 5)
y = torch.randn(3, 4, 5)
arr1 = Array.from_numpy(x.numpy())
arr2 = arr1[1::, ::2, ::3]
arr2 += arr2
arr3 = arr2.sum()
# arr2 = Array.from_numpy(y.numpy())
# arr3 = arr2[1::, ::2, ::3]
# # arr2 = arr1.sum([1])
# arr4 = arr3.sum()
g = MTLGraph(arr3, ctx)
g.compile()
g.forward()
print(arr1)
print(arr2)
# g.backward()
# t1 = x.requires_grad_(True)
# t2 = y.requires_grad_(True)
# t3 = t2[1::, ::2, ::3]
# # t2 = t1.sum(dim=1)
# t4 = t3.sum()
# t4.backward()
# print(arr1.grad)
# print(t1.grad)
# print(arr2.grad)
# print(t2.grad)
