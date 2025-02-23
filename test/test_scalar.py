from python.xavier import Array
from python.metal.graph import MTLGraph
import numpy as np
import time


class TestScalar:
    def run_no_broadcast(self, name: str, op1, op2):
        print(f"{name}:")
        n = np.random.randint(1, 5)
        shape = [np.random.randint(1, 100) for _ in range(n)]
        x1 = np.random.randn(*shape).astype(np.float32)
        x2 = np.random.randn(*shape).astype(np.float32)
        arr1 = Array.from_buffer(x1).reshape(shape)
        arr2 = Array.from_buffer(x2).reshape(shape)
        arr3: Array = op1(arr1, arr2)
        g = MTLGraph(arr3)
        g.compile()
        start = time.time()
        g.forward()
        end = time.time()
        print(f"xv for {shape}: {end - start}")
        x3 = np.frombuffer(arr3, dtype=np.float32)
        start = time.time()
        x4: np.ndarray = op2(x1, x2)
        end = time.time()
        print(f"np for {shape}: {end - start}\n")
        assert tuple(arr3.shape().view()) == x4.shape
        assert np.allclose(x3, x4.flatten(), atol=1e-3, rtol=0)

    def test_add(self):
        def add(x1: Array, x2: Array):
            return x1 + x2

        self.run_no_broadcast("add", add, np.add)

    def test_sub(self):
        def sub(x1: Array, x2: Array):
            return x1 - x2

        self.run_no_broadcast("sub", sub, np.subtract)

    def test_mul(self):
        def mul(x1: Array, x2: Array):
            return x1 * x2

        self.run_no_broadcast("mul", mul, np.multiply)

    def test_div(self):
        def div(x1: Array, x2: Array):
            return x1 / x2

        self.run_no_broadcast("div", div, np.divide)

    def test_exp(self):
        def exp(x: Array):
            return x.exp()

        self.run_no_broadcast("exp", exp, np.exp)
