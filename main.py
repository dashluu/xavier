import time
from python.xavier import MTLGraph, Array, MTLContext, b8, i32, f32
import python.xavier as xv
import numpy as np
import torch

ctx = MTLContext("./xavier/build/backend/metal/kernels.metallib")
shape = [2, 3, 4]
np1 = np.random.uniform(0.1, 2.0, size=shape).astype(np.float32)
np2 = np.random.uniform(0.1, 2.0, size=shape).astype(np.float32)
arr1 = Array.from_buffer(np1).reshape(shape)
arr2 = Array.from_buffer(np2).reshape(shape)
sq1 = arr1.sq()
sqrt1 = sq1.sqrt()
# div1 = arr2 / arr1
# exp1 = div1.exp()
# mul1 = sqrt1 * exp1
# log1 = mul1.log()
# sqrt2 = arr2.sqrt()
# mul2 = arr1 * sqrt2
# sq2 = mul2.sq()
# result = log1 + sq2
result = sqrt1
g = MTLGraph(result, ctx)
g.compile()
g.forward()
g.backward()
# PyTorch implementation
t1 = torch.from_numpy(np1).requires_grad_(True)
t2 = torch.from_numpy(np2).requires_grad_(True)
# Branch 1
tsq1 = t1 * t1
tsq1.retain_grad()
tsqrt1 = torch.sqrt(tsq1)
tsqrt1.retain_grad()
# tdiv1 = t2 / t1
# tdiv1.retain_grad()
# texp1 = torch.exp(tdiv1)
# texp1.retain_grad()
# tmul1 = tsqrt1 * texp1
# tmul1.retain_grad()
# tlog1 = torch.log(tmul1)
# tlog1.retain_grad()
# # Branch 2
# tsqrt2 = torch.sqrt(t2)
# tsqrt2.retain_grad()
# tmul2 = t1 * tsqrt2
# tmul2.retain_grad()
# tsq2 = tmul2 * tmul2
# tsq2.retain_grad()
# Combine branches
# tresult = tlog1 + tsq2
tresult = tsqrt1
tresult.retain_grad()
tsum = tresult.sum()
tsum.backward(retain_graph=True)
print(arr1.grad)
print(t1.grad)
g.backward()
tresult.grad = None
tsum.backward()
print(arr1.grad)
print(t1.grad)
