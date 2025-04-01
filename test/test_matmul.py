import numpy as np
from python.xavier import Array, Shape, MTLContext, MTLGraph


class TestMatmul:
    lib = "./xavier/build/backend/metal/kernels.metallib"

    def test_matmul_2d(self):
        """Test matrix multiplication for 2D arrays"""
        ctx = MTLContext(self.lib)
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
            arr1 = Array.from_numpy(np1)
            arr2 = Array.from_numpy(np2)
            arr3 = arr1 @ arr2
            arr4 = arr3.sum()
            g = MTLGraph(arr4, ctx)
            g.compile()
            g.forward()

            # NumPy comparison
            np3 = np.matmul(np1, np2)
            xv_result = np.frombuffer(arr3, dtype=np.float32).reshape(np3.shape)

            # Verify shape and values
            assert tuple(arr3.view()) == np3.shape
            assert np.allclose(xv_result, np3, atol=1e-3, rtol=0)

    def test_matmul_3d(self):
        """Test matrix multiplication for 3D arrays (batched matmul)"""
        ctx = MTLContext(self.lib)
        print("\nTesting 3D matrix multiplication:")

        # Test cases: [(shape1, shape2, description)]
        test_cases = [
            # Basic batch matmul
            ([4, 2, 3], [4, 3, 4], "Standard batch size"),
            ([1, 2, 3], [1, 3, 4], "Single batch"),
            ([10, 3, 3], [10, 3, 3], "Square matrices batch"),
            # Broadcasting cases
            ([1, 2, 3], [5, 3, 4], "Broadcast first dim"),
            ([5, 2, 3], [1, 3, 4], "Broadcast second dim"),
            ([7, 1, 3], [7, 3, 5], "Batch with singular dimension"),
            # Edge cases
            ([3, 1, 4], [3, 4, 1], "Result has singular dimension"),
            ([2, 5, 1], [2, 1, 3], "Inner dimension is 1"),
            ([1, 1, 1], [1, 1, 1], "All dimensions are 1"),
        ]

        for shape1, shape2, desc in test_cases:
            print(f"\nTesting {desc}:")
            print(f"Shapes: {shape1} @ {shape2}")

            # Generate random matrices
            np1 = np.random.randn(*shape1).astype(np.float32)
            np2 = np.random.randn(*shape2).astype(np.float32)

            # Xavier implementation
            arr1 = Array.from_numpy(np1)
            arr2 = Array.from_numpy(np2)
            arr3 = arr1 @ arr2
            arr4 = arr3.sum()
            g = MTLGraph(arr4, ctx)
            g.compile()
            g.forward()

            # NumPy comparison
            np3 = np.matmul(np1, np2)
            xv_result = arr3.to_numpy()

            # Verify shape and values
            assert tuple(arr3.view()) == np3.shape, f"Shape mismatch: got {arr3.view()}, expected {np3.shape}"
            assert np.allclose(xv_result, np3, atol=1e-3, rtol=0), f"Value mismatch for {desc}"
