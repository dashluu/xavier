import time
from python.xavier import MTLGraph, Array, MTLContext, b8, i32, f32
import numpy as np
import torch

ctx = MTLContext("./xavier/build/backend/metal/kernels.metallib")
x = torch.randn(2, 3, 4, dtype=torch.float32)
arr1 = Array.from_numpy(x.numpy())
arr2 = arr1.max()
g = MTLGraph(arr2, ctx)
g.compile()
g.forward()
t1 = x.requires_grad_(True)
t2 = t1.max()
print(arr2)
print(t2)
