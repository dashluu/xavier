import time
from python.xavier import MTLGraph, Array, MTLContext, b8, i32, f32
import python.xavier as xv
import numpy as np
import torch

ctx = MTLContext("./xavier/build/backend/metal/kernels.metallib")
# Forward: (2,3,4,5) -> permute -> reshape -> matmul -> exp
x = torch.randn(2, 3, 4, 5, dtype=torch.float32)
w = torch.randn(8, 5, 2, dtype=torch.float32)
# Xavier implementation
xv_x = Array.from_numpy(x.numpy())
xv_w = Array.from_numpy(w.numpy())
xv_v = xv_x.permute([0, 2, 1, 3]).reshape([8, 3, 5]) @ xv_w  # (2,4,3,5)  # (8,3,5)  # (8,3,2)
xv_out = xv_v.exp()
g = MTLGraph(xv_out, ctx)
g.compile()
print(g)
g.forward()
g.backward()
x_t = x.requires_grad_(True)
w_t = w.requires_grad_(True)
out_t = x_t.permute(0, 2, 1, 3).reshape(8, 3, 5) @ w_t  # (2,4,3,5)  # (8,3,5)  # (8,3,2)
out_t = torch.exp(out_t)
out_t.sum().backward()
print(xv_w.grad)
print(w_t.grad)
