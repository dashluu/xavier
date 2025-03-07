from python.xavier import Array, MTLGraph, MTLContext
import numpy as np
import time
import torch


class TestBackprop:
    lib = "./xavier/build/backend/metal/kernels.metallib"

    def test_backprop_v1(self):
        ctx = MTLContext(TestBackprop.lib)
        print("backprop 1:")
        n = np.random.randint(1, 5)
        shape = [np.random.randint(1, 100) for _ in range(n)]
        np1 = np.random.randn(*shape).astype(np.float32)
        np2 = np.random.randn(*shape).astype(np.float32)
        arr1 = Array.from_buffer(np1).reshape(shape)
        arr2 = Array.from_buffer(np2).reshape(shape)
        arr3 = arr1 + arr2
        arr4 = arr1 * arr2
        arr5 = arr3 + arr4
        arr6 = arr3 * arr4
        arr7 = arr5 + arr6
        g = MTLGraph(arr7, ctx)
        g.compile()
        g.forward()
        g.backward()
        t1 = torch.from_numpy(np1).requires_grad_(True)
        t2 = torch.from_numpy(np2).requires_grad_(True)
        t3 = t1 + t2
        t3.retain_grad()
        t4 = t1 * t2
        t4.retain_grad()
        t5 = t3 + t4
        t5.retain_grad()
        t6 = t3 * t4
        t6.retain_grad()
        t7 = t5 + t6
        t7.retain_grad()
        t7.sum().backward()
        t8 = torch.frombuffer(arr7, dtype=torch.float32)
        assert torch.allclose(t8, t7.flatten(), atol=1e-3, rtol=0)
        t9 = torch.frombuffer(arr3.grad, dtype=torch.float32)
        assert torch.allclose(t9, t3.grad.flatten(), atol=1e-3, rtol=0)
        t10 = torch.frombuffer(arr4.grad, dtype=torch.float32)
        assert torch.allclose(t10, t4.grad.flatten(), atol=1e-3, rtol=0)
        t11 = torch.frombuffer(arr5.grad, dtype=torch.float32)
        assert torch.allclose(t11, t5.grad.flatten(), atol=1e-3, rtol=0)
        t12 = torch.frombuffer(arr6.grad, dtype=torch.float32)
        assert torch.allclose(t12, t6.grad.flatten(), atol=1e-3, rtol=0)
        t13 = torch.frombuffer(arr7.grad, dtype=torch.float32)
        assert torch.allclose(t13, t7.grad.flatten(), atol=1e-3, rtol=0)
