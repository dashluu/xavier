import numpy as np
from python.xavier import Array, Shape, MTLContext, MTLGraph


class TestCopy:
    lib = "./xavier/build/backend/metal/kernels.metallib"

    def test_contiguous_copy(self):
        ctx = MTLContext(self.lib)
        print("Contiguous copy:")

        # Test cases with different shapes
        test_cases = [
            [10],  # 1D array
            [3, 4],  # 2D array
            [2, 3, 4],  # 3D array
            [2, 3, 4, 5],  # 4D array
        ]

        for shape in test_cases:
            print(f"\nTesting shape: {shape}")
            np1 = np.random.randn(*shape).astype(np.float32)
            arr1 = Array.from_numpy(np1)
            arr2 = arr1.copy()
            arr3 = arr2.sum()
            g = MTLGraph(arr3, ctx)
            g.compile()
            g.forward()
            np2 = np.frombuffer(arr2, dtype=np.float32)
            assert np.allclose(np2, np1.flatten(), atol=1e-6)
            assert tuple(arr2.view()) == np1.shape

    def test_strided_copy(self):
        ctx = MTLContext(self.lib)
        print("Sparse copy:")

        # Test cases with different slicing patterns
        test_cases = [
            # [shape, slices] -> creates tensors with different strides
            ([4, 4], (slice(None, None, 2), slice(None))),  # Skip every other row
            ([4, 6], (slice(None), slice(None, None, 2))),  # Skip every other column
            ([4, 4, 4], (slice(None), slice(None, None, 2), slice(None))),  # Skip in middle dim
            ([6, 6], (slice(None, None, 3), slice(1, None, 2))),  # Complex slicing
        ]

        for shape, slices in test_cases:
            print(f"\nTesting shape: {shape}, slices: {slices}")
            np1 = np.random.randn(*shape).astype(np.float32)
            arr1 = Array.from_numpy(np1)

            # Create non-contiguous array using slicing
            arr2 = arr1[slices]
            expected_shape = arr2.view()

            # Copy the non-contiguous array
            arr3 = arr2.copy()
            arr4 = arr3.sum()

            g = MTLGraph(arr4, ctx)
            g.compile()
            g.forward()

            # Compare with NumPy slicing
            np2 = np1[slices]
            np3 = np.frombuffer(arr3, dtype=np.float32)
            np3 = np3.reshape(expected_shape)

            assert np.allclose(np3, np2, atol=1e-6)
            assert tuple(arr3.view()) == np2.shape
