from python.xavier import Array, MTLGraph, MTLContext
import numpy as np
import time


class TestScalar:
    lib = "./xavier/build/backend/metal/kernels.metallib"

    def binary_no_broadcast(self, name: str, op1, op2):
        ctx = MTLContext(TestScalar.lib)
        print(f"{name}:")
        n = np.random.randint(1, 5)
        shape = [np.random.randint(1, 100) for _ in range(n)]
        np1 = np.random.randn(*shape).astype(np.float32)
        np2 = np.random.randn(*shape).astype(np.float32)
        arr1 = Array.from_buffer(np1).reshape(shape)
        arr2 = Array.from_buffer(np2).reshape(shape)
        arr3: Array = op1(arr1, arr2)
        g = MTLGraph(arr3, ctx)
        g.compile()
        start = time.time()
        g.forward()
        end = time.time()
        print(f"xv for {shape}: {end - start}")
        np3 = np.frombuffer(arr3, dtype=np.float32)
        start = time.time()
        np4: np.ndarray = op2(np1, np2)
        end = time.time()
        print(f"np for {shape}: {end - start}\n")
        assert tuple(arr3.shape().view()) == np4.shape
        assert np.allclose(np3, np4.flatten(), atol=1e-3, rtol=0)

    def unary_no_broadcast(self, name: str, op1, op2):
        ctx = MTLContext(TestScalar.lib)
        print(f"{name}:")
        n = np.random.randint(1, 5)
        shape = [np.random.randint(1, 100) for _ in range(n)]
        np1 = np.random.randn(*shape).astype(np.float32)
        arr1 = Array.from_buffer(np1).reshape(shape)
        arr2: Array = op1(arr1)
        g = MTLGraph(arr2, ctx)
        g.compile()
        start = time.time()
        g.forward()
        end = time.time()
        print(f"xv for {shape}: {end - start}")
        np2 = np.frombuffer(arr2, dtype=np.float32)
        start = time.time()
        np3: np.ndarray = op2(np1)
        end = time.time()
        print(f"np for {shape}: {end - start}\n")
        assert tuple(arr2.shape().view()) == np3.shape
        assert np.allclose(np2, np3.flatten(), atol=1e-3, rtol=0)

    def test_add(self):
        def add(x1: Array, x2: Array):
            return x1 + x2

        self.binary_no_broadcast("add", add, np.add)

    def test_sub(self):
        def sub(x1: Array, x2: Array):
            return x1 - x2

        self.binary_no_broadcast("sub", sub, np.subtract)

    def test_mul(self):
        def mul(x1: Array, x2: Array):
            return x1 * x2

        self.binary_no_broadcast("mul", mul, np.multiply)

    def test_div(self):
        def div(x1: Array, x2: Array):
            return x1 / x2

        self.binary_no_broadcast("div", div, np.divide)

    def test_exp(self):
        def exp(x: Array):
            return x.exp()

        self.unary_no_broadcast("exp", exp, np.exp)

    def test_neg(self):
        def neg(x: Array):
            return -x

        self.unary_no_broadcast("neg", neg, np.negative)

    def test_log(self):
        ctx = MTLContext(TestScalar.lib)
        print("log:")
        n = np.random.randint(1, 5)
        shape = [np.random.randint(1, 100) for _ in range(n)]
        np1 = np.abs(np.random.randn(*shape).astype(np.float32))
        arr1 = Array.from_buffer(np1).reshape(shape)
        arr2 = arr1.log()
        g = MTLGraph(arr2, ctx)
        g.compile()
        start = time.time()
        g.forward()
        end = time.time()
        print(f"xv for {shape}: {end - start}")
        np2 = np.frombuffer(arr2, dtype=np.float32)
        start = time.time()
        np3 = np.log(np1)
        end = time.time()
        print(f"np for {shape}: {end - start}\n")
        assert tuple(arr2.shape().view()) == np3.shape
        assert np.allclose(np2, np3.flatten(), atol=1e-3, rtol=0)

    def test_recip(self):
        ctx = MTLContext(TestScalar.lib)
        print("recip:")
        n = np.random.randint(1, 5)
        shape = [np.random.randint(1, 100) for _ in range(n)]
        # Prevent the values from being 0 to avoid division by zero
        np1 = np.abs(np.random.randn(*shape).astype(np.float32)) + 1
        arr1 = Array.from_buffer(np1).reshape(shape)
        arr2: Array = arr1.recip()
        g = MTLGraph(arr2, ctx)
        g.compile()
        start = time.time()
        g.forward()
        end = time.time()
        print(f"xv for {shape}: {end - start}")
        np2 = np.frombuffer(arr2, dtype=np.float32)
        start = time.time()
        np3 = np.reciprocal(np1)
        end = time.time()
        print(f"np for {shape}: {end - start}\n")
        assert tuple(arr2.shape().view()) == np3.shape
        assert np.allclose(np2, np3.flatten(), atol=1e-3, rtol=0)
