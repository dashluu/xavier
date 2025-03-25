import time
from python.xavier import MTLGraph, Array, MTLContext, b8, i32, f32
import python.xavier as xv
import numpy as np
import torch

ctx = MTLContext("./xavier/build/backend/metal/kernels.metallib")
shape1 = [2, 3]
shape2 = [3, 4]
np1 = np.random.randn(*shape1).astype(np.float32)
np2 = np.random.randn(*shape2).astype(np.float32)
np3 = np1 @ np2
arr1 = Array.from_numpy(np1)
arr2 = Array.from_numpy(np2)
arr3 = arr1 @ arr2
g = MTLGraph(arr3, ctx)
g.compile()
g.forward()
np4 = arr3.to_numpy()
print(np1)
print(np2)
print(np3)
print(np4)
# s1 = [2, 2, 3]
# s2 = [2, 3, 4]
# np1 = np.random.randn(*s1).astype(np.float32)
# np2 = np.random.randn(*s2).astype(np.float32)
# # Xavier implementation
# arr1 = Array.from_buffer(np1).reshape(s1)
# arr2 = Array.from_buffer(np2).reshape(s2)
# arr3 = arr1 @ arr2  # Use matmul operator
# g = MTLGraph(arr3, ctx)
# g.compile()
# g.forward()
# # NumPy comparison
# np3 = np.matmul(np1, np2)
# print(arr3)
# print(np3)
