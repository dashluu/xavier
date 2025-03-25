from python.xavier import Array, MTLGraph, MTLContext
import numpy as np


class TestTransform:
    lib = "./xavier/build/backend/metal/kernels.metallib"

    def test_slice_v1(self):
        ctx = MTLContext(TestTransform.lib)
        print("slice 1:")
        s = [np.random.randint(1, 50) for _ in range(3)]
        a = np.random.randn(*s).astype(np.float32)
        arr1 = Array.from_numpy(a)
        arr2 = arr1[::, ::, ::].as_contiguous()
        g = MTLGraph(arr2, ctx)
        g.compile()
        g.forward()
        np1 = arr2.to_numpy()
        np2 = a[::, ::, ::]
        assert np.allclose(np1, np2, atol=1e-3, rtol=0)

    def test_slice_v2(self):
        ctx = MTLContext(TestTransform.lib)
        print("slice 2:")
        s = [np.random.randint(4, 50) for _ in range(4)]
        a = np.random.randn(*s).astype(np.float32)
        arr1 = Array.from_numpy(a)
        arr2 = arr1[1::4, :3:2, 2::3].as_contiguous()
        g = MTLGraph(arr2, ctx)
        g.compile()
        g.forward()
        np1 = arr2.to_numpy()
        np2 = a[1::4, :3:2, 2::3]
        assert np.allclose(np1, np2, atol=1e-3, rtol=0)

    def test_slice_v3(self):
        ctx = MTLContext(TestTransform.lib)
        print("slice 3:")
        s = [np.random.randint(4, 50) for _ in range(4)]
        a = np.random.randn(*s).astype(np.float32)
        arr1 = Array.from_numpy(a)
        arr2 = arr1[1::, ::2, 3:0:-2].as_contiguous()
        g = MTLGraph(arr2, ctx)
        g.compile()
        g.forward()
        np1 = arr2.to_numpy()
        np2 = a[1::, ::2, 3:0:-2]
        assert np.allclose(np1, np2, atol=1e-3, rtol=0)

    def test_slice_v4(self):
        ctx = MTLContext(TestTransform.lib)
        print("slice 4:")
        s = [np.random.randint(10, 50) for _ in range(4)]
        a = np.random.randn(*s).astype(np.float32)
        arr1 = Array.from_numpy(a)
        arr2 = arr1[1:0:-4, 9:3:-2, 2::3].as_contiguous()
        g = MTLGraph(arr2, ctx)
        g.compile()
        g.forward()
        np1 = arr2.to_numpy()
        np2 = a[1:0:-4, 9:3:-2, 2::3]
        assert np.allclose(np1, np2, atol=1e-3, rtol=0)

    def test_transpose_start(self):
        ctx = MTLContext(TestTransform.lib)
        print("transpose at the start:")
        s = [np.random.randint(3, 10) for _ in range(4)]
        a = np.random.randn(*s).astype(np.float32)
        arr1 = Array.from_numpy(a)
        # Xavier transpose
        arr2 = arr1.T(0, 2)
        g = MTLGraph(arr2, ctx)
        g.compile()
        g.forward()
        np1 = arr2.to_numpy()
        # NumPy transpose - create same permutation as Xavier
        order = list(range(len(s)))  # [0,1,2,3]
        # Reverse order from start_dim to end_dim
        order[0 : 2 + 1] = order[0 : 2 + 1][::-1]  # [2,1,0,3]
        np2 = np.transpose(a, order)
        assert np.allclose(np1, np2, atol=1e-3, rtol=0)

    def test_transpose_mid(self):
        ctx = MTLContext(TestTransform.lib)
        print("transpose in the middle:")
        s = [np.random.randint(3, 10) for _ in range(6)]
        a = np.random.randn(*s).astype(np.float32)
        arr1 = Array.from_numpy(a)
        # Xavier transpose
        arr2 = arr1.T(1, -2)
        g = MTLGraph(arr2, ctx)
        g.compile()
        g.forward()
        np1 = arr2.to_numpy()
        # NumPy transpose - create same permutation as Xavier
        order = list(range(len(s)))  # [0,1,2,3]
        # Reverse order from start_dim to end_dim
        order[1:-1] = order[1:-1][::-1]  # [0,3,2,1]
        np2 = np.transpose(a, order)
        assert np.allclose(np1, np2, atol=1e-3, rtol=0)

    def test_transpose_end(self):
        ctx = MTLContext(TestTransform.lib)
        print("transpose at the end:")
        s = [np.random.randint(3, 10) for _ in range(5)]
        a = np.random.randn(*s).astype(np.float32)
        arr1 = Array.from_numpy(a)
        # Xavier transpose
        arr2 = arr1.T(-3, -1)
        g = MTLGraph(arr2, ctx)
        g.compile()
        g.forward()
        np1 = arr2.to_numpy()
        # NumPy transpose - create same permutation as Xavier
        order = list(range(len(s)))  # [0,1,2,3]
        # Reverse order from start_dim to end_dim
        order[-3:] = order[-3:][::-1]  # [0,3,2,1]
        np2 = np.transpose(a, order)
        assert np.allclose(np1, np2, atol=1e-3, rtol=0)

    def test_permute(self):
        ctx = MTLContext(TestTransform.lib)
        print("\nTesting permute operations:")

        # Test cases: [(shape, permutation)]
        test_cases = [
            # Basic permutations
            ([2, 3, 4], [2, 0, 1]),  # 3D rotation
            ([2, 3, 4, 5], [3, 2, 1, 0]),  # Complete reverse
            ([2, 3, 4, 5], [0, 2, 1, 3]),  # Middle swap
            # Edge cases
            ([1, 2, 3], [2, 1, 0]),  # With dimension size 1
            ([5, 1, 1, 4], [0, 2, 1, 3]),  # Multiple size-1 dimensions
            ([2, 3], [1, 0]),  # 2D transpose
        ]

        for shape, perm in test_cases:
            print(f"\nTesting shape {shape} with permutation {perm}")

            # Create test data
            a = np.random.randn(*shape).astype(np.float32)
            arr1 = Array.from_numpy(a)

            # Xavier permute
            arr2 = arr1.permute(perm)
            g = MTLGraph(arr2, ctx)
            g.compile()
            g.forward()
            xv_result = arr2.to_numpy()

            # NumPy permute
            np_result = np.transpose(a, perm)

            # Verify results
            assert np.allclose(
                xv_result, np_result, atol=1e-3, rtol=0
            ), f"Permute failed for shape {shape} with perm {perm}"
            assert (
                xv_result.shape == np_result.shape
            ), f"Shape mismatch: got {xv_result.shape}, expected {np_result.shape}"

    def test_flatten(self):
        ctx = MTLContext(TestTransform.lib)
        print("\nTesting flatten operations:")

        # Test cases: [(shape, start_dim, end_dim, expected_shape)]
        test_cases = [
            # Basic flattening
            ([2, 3, 4], 0, -1, [24]),  # Flatten all
            ([2, 3, 4, 5], 1, 2, [2, 12, 5]),  # Middle flatten
            ([2, 3, 4, 5], 0, 1, [6, 4, 5]),  # Start flatten
            ([2, 3, 4, 5], -2, -1, [2, 3, 20]),  # End flatten
            # Edge cases
            ([1, 2, 3, 4], 1, 3, [1, 24]),  # With leading 1
            ([2, 1, 3, 1], 1, 2, [2, 3, 1]),  # With middle 1s
            ([5], 0, 0, [5]),  # Single dimension
        ]

        for shape, start, end, expected in test_cases:
            print(f"\nTesting shape {shape} flatten({start},{end})")

            # Create test data
            a = np.random.randn(*shape).astype(np.float32)
            arr1 = Array.from_numpy(a)

            # Xavier flatten
            arr2 = arr1.flatten(start, end)
            g = MTLGraph(arr2, ctx)
            g.compile()
            g.forward()
            xv_result = arr2.to_numpy()

            # NumPy flatten - reshape to match expected shape
            np_result = a.reshape(expected)

            # Verify results
            assert np.allclose(
                xv_result, np_result, atol=1e-3, rtol=0
            ), f"Flatten failed for shape {shape} with dims {start},{end}"
            assert xv_result.shape == tuple(expected), f"Shape mismatch: got {xv_result.shape}, expected {expected}"
