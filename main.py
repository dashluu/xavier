import time
from python.xavier import MTLGraph, Array, MTLContext, b8, i32, f32
import numpy as np
import torch

ctx = MTLContext("./xavier/build/backend/metal/kernels.metallib")
x = torch.arange(0, 60, dtype=torch.float32).reshape(3, 4, 5)
arr1 = Array.from_numpy(x.numpy().copy())
arr2 = arr1[1::, ::2, ::3]
arr2 += arr2  # Double the values
g = MTLGraph(arr2.sum(), ctx)
g.compile()
g.forward()
t = x[1::, ::2, ::3] * 2
result = arr2.numpy()
print(result)
print(t.numpy())
assert np.allclose(result, t.numpy())
