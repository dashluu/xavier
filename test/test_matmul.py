import numpy as np
from python.xavier import Array, Shape, MTLContext, MTLGraph


class TestMatmul:
    lib = "./xavier/build/backend/metal/kernels.metallib"

    def test_matmul_2d(self):
        """Test matrix multiplication for 2D arrays"""
        ctx = MTLContext(TestMatmul.lib)
        print("matmul 2d:")

        # Test cases: [(shape1, shape2)]
        test_cases = [
            ([2, 3], [3, 4]),  # Basic matrix multiplication
            ([1, 4], [4, 5]),  # Single row matrix
            ([3, 2], [2, 1]),  # Result is a column matrix
            ([5, 5], [5, 5]),  # Square matrices
            ([1, 1], [1, 1]),  # 1x1 matrices
        ]

        for shape1, shape2 in test_cases:
            print(f"\nTesting shapes: {shape1} @ {shape2}")
            # Generate random matrices
            np1 = np.random.randn(*shape1).astype(np.float32)
            np2 = np.random.randn(*shape2).astype(np.float32)

            # Xavier implementation
            arr1 = Array.from_buffer(np1).reshape(shape1)
            arr2 = Array.from_buffer(np2).reshape(shape2)
            arr3 = arr1 @ arr2  # Use matmul operator
            g = MTLGraph(arr3, ctx)
            g.compile()
            g.forward()

            # NumPy comparison
            np3 = np.matmul(np1, np2)
            xv_result = np.frombuffer(arr3, dtype=np.float32).reshape(np3.shape)

            # Verify shape and values
            assert tuple(arr3.shape().view()) == np3.shape
            assert np.allclose(xv_result, np3, atol=1e-3, rtol=0)
