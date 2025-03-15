from __future__ import annotations
from python.xavier import Array, MTLGraph, MTLContext
import python.xavier as xv
import numpy as np


def randn(shape) -> np.ndarray:
    return np.random.randn(*shape).astype(np.float32)


def nonzero_randn(shape) -> np.ndarray:
    arr = randn(shape)
    # Replace zeros with small random values
    zero_mask = arr == 0
    arr[zero_mask] = np.random.uniform(0.1, 1.0, size=np.count_nonzero(zero_mask))
    return arr


def pos_randn(shape) -> np.ndarray:
    arr = nonzero_randn(shape)
    return np.abs(arr)


class TestScalar:
    lib = "./xavier/build/backend/metal/kernels.metallib"

    def binary_no_broadcast(self, name: str, op1, op2, gen=randn):
        ctx = MTLContext(TestScalar.lib)
        print(f"{name}:")
        n = np.random.randint(1, 5)
        shape = [np.random.randint(1, 100) for _ in range(n)]
        np1 = gen(shape)
        np2 = gen(shape)
        arr1 = Array.from_buffer(np1).reshape(shape)
        arr2 = Array.from_buffer(np2).reshape(shape)
        arr3: Array = op1(arr1, arr2)
        g = MTLGraph(arr3, ctx)
        g.compile()
        g.forward()
        np3 = np.frombuffer(arr3, dtype=np.float32)
        np4: np.ndarray = op2(np1, np2)
        assert tuple(arr3.shape().view()) == np4.shape
        assert np.allclose(np3, np4.flatten(), atol=1e-3, rtol=0)

    def unary_no_broadcast(self, name: str, op1, op2, gen=randn):
        ctx = MTLContext(TestScalar.lib)
        print(f"{name}:")
        n = np.random.randint(1, 5)
        shape = [np.random.randint(1, 100) for _ in range(n)]
        np1 = gen(shape)
        arr1 = Array.from_buffer(np1).reshape(shape)
        arr2: Array = op1(arr1)
        g = MTLGraph(arr2, ctx)
        g.compile()
        g.forward()
        np2 = np.frombuffer(arr2, dtype=np.float32)
        np3: np.ndarray = op2(np1)
        assert tuple(arr2.shape().view()) == np3.shape
        assert np.allclose(np2, np3.flatten(), atol=1e-3, rtol=0)

    def binary_with_broadcast(self, name: str, op1, op2, gen=randn):
        ctx = MTLContext(TestScalar.lib)
        print(f"{name} with broadcast:")
        # Test cases with different broadcasting scenarios
        test_cases = [
            # [shape1, shape2, result_shape]
            ([2, 1, 4], [3, 4], [2, 3, 4]),  # Left broadcast
            ([1, 5], [2, 1, 5], [2, 1, 5]),  # Right broadcast
            ([3, 1, 1], [1, 4, 5], [3, 4, 5]),  # Both broadcast
            ([1], [2, 3, 4], [2, 3, 4]),  # Scalar to array
            ([2, 3, 4], [1], [2, 3, 4]),  # Array to scalar
        ]
        for shape1, shape2, expected_shape in test_cases:
            print(f"\nTesting shapes: {shape1}, {shape2} -> {expected_shape}")
            np1 = gen(shape1)
            np2 = gen(shape2)
            arr1 = Array.from_buffer(np1).reshape(shape1)
            arr2 = Array.from_buffer(np2).reshape(shape2)
            arr3: Array = op1(arr1, arr2)
            g = MTLGraph(arr3, ctx)
            g.compile()
            g.forward()
            np3 = np.frombuffer(arr3, dtype=np.float32)
            np4: np.ndarray = op2(np1, np2)
            assert tuple(arr3.shape().view()) == np4.shape
            assert np.allclose(np3, np4.flatten(), atol=1e-3, rtol=0)

    def unary_with_slicing(self, name: str, op1, op2, gen=randn):
        ctx = MTLContext(TestScalar.lib)
        print(f"{name} with slicing:")

        # Test cases with different slicing patterns
        test_cases = [
            # [shape, slices] -> creates non-contiguous tensors
            ([4, 4], (slice(None, None, 2), slice(None))),  # Skip every other row
            ([4, 6], (slice(None), slice(None, None, 2))),  # Skip every other column
            ([4, 4, 4], (slice(None), slice(1, 3), slice(None))),  # Middle slice
            ([6, 6], (slice(None, None, 3), slice(1, None, 2))),  # Complex slicing
        ]

        for shape, slices in test_cases:
            print(f"\nTesting shape: {shape}, slices: {slices}")
            np1 = gen(shape)
            arr1 = Array.from_buffer(np1).reshape(shape)

            # Create non-contiguous array using slicing
            arr2 = arr1[slices]
            arr3: Array = op1(arr2)  # Apply unary operation
            g = MTLGraph(arr3, ctx)
            g.compile()
            g.forward()

            # Compare with NumPy
            np2 = np1[slices]  # Apply same slicing
            np3: np.ndarray = op2(np2)  # Apply same operation
            np_arr3 = np.frombuffer(arr3, dtype=np.float32)
            np_arr3 = np_arr3.reshape(np3.shape)
            assert np.allclose(np_arr3, np3, atol=1e-3, rtol=0)
            assert tuple(arr3.shape().view()) == np3.shape

    def binary_inplace(self, name: str, op1, op2, gen=randn):
        ctx = MTLContext(TestScalar.lib)
        print(f"{name} inplace:")
        n = np.random.randint(1, 5)
        shape = [np.random.randint(1, 100) for _ in range(n)]

        # Generate inputs
        np1: np.ndarray = gen(shape)
        np2: np.ndarray = gen(shape)
        np1_copy = np1.copy()  # Keep copy for numpy comparison

        # Create arrays
        arr1 = Array.from_buffer(np1).reshape(shape)
        arr2 = Array.from_buffer(np2).reshape(shape)

        # Apply inplace operation
        arr1 = op1(arr1, arr2)  # arr1 += arr2, etc.
        g = MTLGraph(arr1, ctx)  # Use arr1 since it was modified in-place
        g.compile()
        g.forward()

        # Compare with NumPy
        np1_copy: np.ndarray = op2(np1_copy, np2)  # np1_copy += np2, etc.
        np_result = np.frombuffer(arr1, dtype=np.float32)
        assert tuple(arr1.shape().view()) == np1_copy.shape
        assert np.allclose(np_result, np1_copy.flatten(), atol=1e-3, rtol=0)

    def binary_inplace_broadcast(self, name: str, op1, op2, gen=randn):
        ctx = MTLContext(TestScalar.lib)
        print(f"{name} inplace broadcast:")

        test_cases = [
            # [lhs_shape, rhs_shape] -> result shape will be lhs_shape
            ([2, 3, 4], [4]),  # Broadcast scalar to 3D
            ([3, 4, 5], [1, 5]),  # Broadcast from 2D to 3D
            ([2, 4, 6], [4, 1]),  # Broadcast with ones
            ([5, 5, 5], [1, 5, 1]),  # Broadcast with ones in multiple dims
            ([4, 3, 2], [3, 1]),  # Partial broadcast with ones
        ]

        for lhs_shape, rhs_shape in test_cases:
            print(f"\nTesting: {lhs_shape} @= {rhs_shape}")

            # Generate inputs
            np1: np.ndarray = gen(lhs_shape)
            np2: np.ndarray = gen(rhs_shape)
            np1_copy = np1.copy()

            # Create arrays
            arr1 = Array.from_buffer(np1).reshape(lhs_shape)
            arr2 = Array.from_buffer(np2).reshape(rhs_shape)

            # Apply inplace operation
            arr1: Array = op1(arr1, arr2)
            g = MTLGraph(arr1, ctx)
            g.compile()
            g.forward()

            # Compare with NumPy
            np1_copy: np.ndarray = op2(np1_copy, np2)
            xv_result = np.frombuffer(arr1, dtype=np.float32).reshape(lhs_shape)
            assert np.allclose(xv_result, np1_copy, atol=1e-3, rtol=0)
            assert tuple(arr1.shape().view()) == np1_copy.shape

    def unary_inplace(self, name: str, op1, op2, gen=randn):
        ctx = MTLContext(TestScalar.lib)
        print(f"{name} inplace:")

        # Test different shapes
        test_cases = [
            [5],  # 1D
            [2, 3],  # 2D
            [2, 3, 4],  # 3D
            [1, 2, 3, 4],  # 4D with leading 1
            [5, 1, 4],  # 3D with middle 1
        ]

        for shape in test_cases:
            print(f"\nTesting shape: {shape}")

            # Generate input
            np1: np.ndarray = gen(shape)
            np1_copy = np1.copy()

            # Create array
            arr1 = Array.from_buffer(np1).reshape(shape)

            # Apply inplace operation
            arr1: Array = op1(arr1)
            g = MTLGraph(arr1, ctx)
            g.compile()
            g.forward()

            # Compare with NumPy
            np2 = op2(np1_copy)
            xv_result = np.frombuffer(arr1, dtype=np.float32).reshape(shape)
            assert np.allclose(xv_result, np2, atol=1e-3, rtol=0)
            assert tuple(arr1.shape().view()) == np2.shape

    def test_add(self):
        self.binary_no_broadcast("add", xv.add, np.add)

    def test_sub(self):
        self.binary_no_broadcast("sub", xv.sub, np.subtract)

    def test_mul(self):
        self.binary_no_broadcast("mul", xv.mul, np.multiply)

    def test_div(self):
        self.binary_no_broadcast("div", xv.div, np.divide, gen=nonzero_randn)

    def test_add_broadcast(self):
        self.binary_with_broadcast("add", xv.add, np.add)

    def test_mul_broadcast(self):
        self.binary_with_broadcast("mul", xv.mul, np.multiply)

    def test_div_broadcast(self):
        self.binary_with_broadcast("div", xv.div, np.divide, gen=nonzero_randn)

    def test_sub_broadcast(self):
        self.binary_with_broadcast("sub", xv.sub, np.subtract)

    def test_exp(self):
        self.unary_no_broadcast("exp", xv.exp, np.exp)

    def test_neg(self):
        self.unary_no_broadcast("neg", xv.neg, np.negative)

    def test_log(self):
        self.unary_no_broadcast("log", xv.log, np.log, gen=pos_randn)

    def test_recip(self):
        self.unary_no_broadcast("recip", xv.recip, np.reciprocal, gen=nonzero_randn)

    def test_exp_sliced(self):
        self.unary_with_slicing("exp", xv.exp, np.exp)

    def test_neg_sliced(self):
        self.unary_with_slicing("neg", xv.neg, np.negative)

    def test_log_sliced(self):
        self.unary_with_slicing("log", xv.log, np.log, gen=pos_randn)

    def test_recip_sliced(self):
        def recip(x: Array):
            return x.recip()

        self.unary_with_slicing("recip", recip, np.reciprocal, gen=nonzero_randn)

    def test_iadd(self):
        def iadd(x1: Array, x2: Array):
            x1 += x2
            return x1

        def numpy_iadd(x1: np.ndarray, x2: np.ndarray):
            x1 += x2
            return x1

        self.binary_inplace("iadd", iadd, numpy_iadd)

    def test_isub(self):
        def isub(x1: Array, x2: Array):
            x1 -= x2
            return x1

        def numpy_isub(x1: np.ndarray, x2: np.ndarray):
            x1 -= x2
            return x1

        self.binary_inplace("isub", isub, numpy_isub)

    def test_imul(self):
        def imul(x1: Array, x2: Array):
            x1 *= x2
            return x1

        def numpy_imul(x1: np.ndarray, x2: np.ndarray):
            x1 *= x2
            return x1

        self.binary_inplace("imul", imul, numpy_imul)

    def test_idiv(self):
        def idiv(x1: Array, x2: Array):
            x1 /= x2
            return x1

        def numpy_idiv(x1: np.ndarray, x2: np.ndarray):
            x1 /= x2
            return x1

        self.binary_inplace("idiv", idiv, numpy_idiv, gen=nonzero_randn)

    def test_iadd_broadcast(self):
        def iadd(x1: Array, x2: Array):
            x1 += x2
            return x1

        self.binary_inplace_broadcast("iadd", iadd, np.add)

    def test_isub_broadcast(self):
        def isub(x1: Array, x2: Array):
            x1 -= x2
            return x1

        self.binary_inplace_broadcast("isub", isub, np.subtract)

    def test_imul_broadcast(self):
        def imul(x1: Array, x2: Array):
            x1 *= x2
            return x1

        self.binary_inplace_broadcast("imul", imul, np.multiply)

    def test_idiv_broadcast(self):
        def idiv(x1: Array, x2: Array):
            x1 /= x2
            return x1

        self.binary_inplace_broadcast("idiv", idiv, np.divide, gen=nonzero_randn)

    def test_exp_inplace(self):
        def exp_inplace(x: Array):
            return x.exp(in_place=True)

        self.unary_inplace("exp", exp_inplace, np.exp)

    def test_sqrt_inplace(self):
        def sqrt_inplace(x: Array):
            return x.sqrt(in_place=True)

        self.unary_inplace("sqrt", sqrt_inplace, np.sqrt, gen=pos_randn)

    def test_neg_inplace(self):
        def neg_inplace(x: Array):
            return x.neg(in_place=True)

        self.unary_inplace("neg", neg_inplace, np.negative)

    def test_recip_inplace(self):
        def recip_inplace(x: Array):
            return x.recip(in_place=True)

        self.unary_inplace("recip", recip_inplace, np.reciprocal, gen=nonzero_randn)

    def test_log_inplace(self):
        def log_inplace(x: Array):
            return x.log(in_place=True)

        self.unary_inplace("log", log_inplace, np.log, gen=pos_randn)

    def test_const_mul_left(self):
        """Test multiplication with constant on the left side (c * arr)"""
        ctx = MTLContext(TestScalar.lib)
        print("const mul left:")

        # Test cases: [(constant, shape)]
        test_cases = [
            (2.0, [3, 4]),  # Basic 2D
            (-0.5, [2, 3, 4]),  # 3D with negative constant
            (1.5, [1, 2, 3, 4]),  # 4D
            (0.0, [5]),  # Zero constant
            (3.14, [1]),  # Scalar array
        ]

        for const, shape in test_cases:
            print(f"\nTesting const {const} * shape {shape}")
            np1 = randn(shape)
            arr1 = Array.from_buffer(np1).reshape(shape)

            # Xavier implementation
            arr2 = const * arr1  # Constant multiplication
            g = MTLGraph(arr2, ctx)
            g.compile()
            g.forward()

            # NumPy comparison
            np2 = const * np1
            np_result = np.frombuffer(arr2, dtype=np.float32)

            assert tuple(arr2.shape().view()) == np2.shape
            assert np.allclose(np_result, np2.flatten(), atol=1e-3, rtol=0)

    def test_const_mul_right(self):
        """Test multiplication with constant on the right side (arr * c)"""
        ctx = MTLContext(TestScalar.lib)
        print("const mul right:")

        # Test cases: [(shape, constant)]
        test_cases = [
            ([3, 4], 2.0),  # Basic 2D
            ([2, 3, 4], -0.5),  # 3D with negative constant
            ([1, 2, 3, 4], 1.5),  # 4D
            ([5], 0.0),  # Zero constant
            ([1], 3.14),  # Scalar array
        ]

        for shape, const in test_cases:
            print(f"\nTesting shape {shape} * const {const}")
            np1 = randn(shape)
            arr1 = Array.from_buffer(np1).reshape(shape)

            # Xavier implementation
            arr2 = arr1 * const  # Constant multiplication
            g = MTLGraph(arr2, ctx)
            g.compile()
            g.forward()

            # NumPy comparison
            np2 = np1 * const
            np_result = np.frombuffer(arr2, dtype=np.float32)

            assert tuple(arr2.shape().view()) == np2.shape
            assert np.allclose(np_result, np2.flatten(), atol=1e-3, rtol=0)
