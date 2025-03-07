from python.xavier import Array, MTLGraph, MTLContext
import numpy as np


class TestSlice:
    lib = "./xavier/build/backend/metal/kernels.metallib"

    def test_slice_v1(self):
        ctx = MTLContext(TestSlice.lib)
        print("slice 1:")
        s = [np.random.randint(1, 50) for _ in range(3)]
        a = np.random.randn(*s).astype(np.float32)
        arr1 = Array.from_buffer(a).reshape(s)
        arr2 = arr1[::, ::, ::].as_contiguous()
        g = MTLGraph(arr2, ctx)
        g.compile()
        g.forward()
        np1 = np.frombuffer(arr2, dtype=np.float32).reshape(arr2.shape().view())
        np2 = a[::, ::, ::]
        assert np.allclose(np1, np2, atol=1e-3, rtol=0)

    def test_slice_v2(self):
        ctx = MTLContext(TestSlice.lib)
        print("slice 2:")
        s = [np.random.randint(4, 50) for _ in range(4)]
        a = np.random.randn(*s).astype(np.float32)
        arr1 = Array.from_buffer(a).reshape(s)
        arr2 = arr1[1::4, :3:2, 2::3].as_contiguous()
        g = MTLGraph(arr2, ctx)
        g.compile()
        g.forward()
        np1 = np.frombuffer(arr2, dtype=np.float32).reshape(arr2.shape().view())
        np2 = a[1::4, :3:2, 2::3]
        assert np.allclose(np1, np2, atol=1e-3, rtol=0)

    def test_slice_v3(self):
        ctx = MTLContext(TestSlice.lib)
        print("slice 3:")
        s = [np.random.randint(4, 50) for _ in range(4)]
        a = np.random.randn(*s).astype(np.float32)
        arr1 = Array.from_buffer(a).reshape(s)
        arr2 = arr1[1::, ::2, 3:0:-2].as_contiguous()
        g = MTLGraph(arr2, ctx)
        g.compile()
        g.forward()
        np1 = np.frombuffer(arr2, dtype=np.float32).reshape(arr2.shape().view())
        np2 = a[1::, ::2, 3:0:-2]
        assert np.allclose(np1, np2, atol=1e-3, rtol=0)

    def test_slice_v4(self):
        ctx = MTLContext(TestSlice.lib)
        print("slice 4:")
        s = [np.random.randint(10, 50) for _ in range(4)]
        a = np.random.randn(*s).astype(np.float32)
        arr1 = Array.from_buffer(a).reshape(s)
        arr2 = arr1[1:0:-4, 9:3:-2, 2::3].as_contiguous()
        g = MTLGraph(arr2, ctx)
        g.compile()
        g.forward()
        np1 = np.frombuffer(arr2, dtype=np.float32).reshape(arr2.shape().view())
        np2 = a[1:0:-4, 9:3:-2, 2::3]
        assert np.allclose(np1, np2, atol=1e-3, rtol=0)
