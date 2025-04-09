import torch
import numpy as np
from python.xavier import MTLGraph, Array, MTLContext, b8, i32, f32


class TestReduction:
    def setup_method(self):
        self.ctx = MTLContext("./xavier/build/backend/metal/kernels.metallib")

    def test_basic_sum_reduction(self):
        """Test basic sum reduction without specified dimensions"""
        print("\nTesting basic sum reduction:")

        # Create test data
        test_cases = [[2, 3, 4], [3, 4, 5], [31, 67, 18, 17], [1027, 64, 32], [5674, 3289], [1, 1]]

        for shape in test_cases:
            x = torch.randn(shape, dtype=torch.float32)

            # Xavier implementation
            arr1 = Array.from_numpy(x.numpy())
            arr2 = arr1.sum()
            g = MTLGraph(arr2, self.ctx)
            g.compile()
            g.forward()

            # PyTorch comparison
            expected = np.array([x.sum()])
            # Use relative error here to measure the difference
            err = np.abs(arr2.numpy() - expected) / max(np.abs(arr2.numpy()), np.abs(expected))
            assert err <= 1e-3

    def test_col_sum_reduction(self):
        """Test sum reduction along the column dimension"""
        print("\nTesting sum reduction along the column dimension:")

        # Test cases with different dimensions
        # shapes = [(2, 3, 4), (4, 5), (3, 4, 5, 6)]
        # dims_list = [[0], [1], [0, 2], [], [0, 1, 2]]
        shapes = [(2, 3), (19, 29), (37, 32), (47, 7), (297, 101), (5674, 3289), (256, 1), (1, 1), (1, 997)]
        dims_list = [[1], [1], [1], [1]]

        for shape in shapes:
            x = torch.randn(*shape, dtype=torch.float32)
            arr1 = Array.from_numpy(x.numpy())

            for dims in dims_list:
                if len(dims) > len(shape):
                    continue

                # Xavier implementation
                arr2 = arr1.sum(dims)
                arr3 = arr2.sum()
                g = MTLGraph(arr3, self.ctx)
                g.compile()
                g.forward()

                # PyTorch comparison
                expected = x
                if dims:  # If dims is not empty
                    expected = x.sum(dim=dims).unsqueeze(dim=-1)
                else:  # If dims is empty, sum all
                    expected = x.sum()

                print(shape)
                print(arr2.numpy().flatten())
                print(expected.flatten())
                assert np.allclose(arr2.numpy(), expected.numpy(), atol=1e-3, rtol=0)

    def test_max_reduction(self):
        """Test max reduction with various scenarios"""
        print("\nTesting max reduction:")

        # Test data with known max values
        x = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=torch.float32)

        arr1 = Array.from_numpy(x.numpy())

        # Test max reduction along different dimensions
        test_dims = [
            ([0], "max along batch"),
            ([1], "max along rows"),
            ([2], "max along columns"),
            ([0, 1], "max along batch and rows"),
            ([], "max all elements"),
        ]

        for dims, desc in test_dims:
            print(f"Testing {desc}")

            # Xavier implementation
            arr2 = arr1.max(dims)
            g = MTLGraph(arr2, self.ctx)
            g.compile()
            g.forward()

            # PyTorch comparison
            expected = x.max(dim=dims[0])[0] if len(dims) == 1 else x.max()
            if len(dims) > 1:
                for d in dims[1:]:
                    expected = expected.max(dim=d - len(dims))[0]

            assert np.allclose(arr2.numpy(), expected.numpy())

    def test_reduction_edge_cases(self):
        """Test reduction operations with edge cases"""
        print("\nTesting reduction edge cases:")

        # Test cases with special values
        edge_cases = [
            torch.tensor([0, 0, 0], dtype=torch.float32),  # All zeros
            torch.tensor([1, 1, 1], dtype=torch.float32),  # All ones
            torch.tensor([-1, 0, 1], dtype=torch.float32),  # Mixed values
            torch.tensor([float("inf")], dtype=torch.float32),  # Infinity
            torch.tensor([float("-inf")], dtype=torch.float32),  # Negative infinity
        ]

        for x in edge_cases:
            # Xavier implementation
            arr1 = Array.from_numpy(x.numpy())
            arr2_sum = arr1.sum()
            arr2_max = arr1.max()

            g_sum = MTLGraph(arr2_sum, self.ctx)
            g_max = MTLGraph(arr2_max, self.ctx)

            g_sum.compile()
            g_max.compile()

            g_sum.forward()
            g_max.forward()

            # PyTorch comparison
            expected_sum = x.sum()
            expected_max = x.max()

            assert np.allclose(arr2_sum.numpy(), expected_sum.numpy())
            assert np.allclose(arr2_max.numpy(), expected_max.numpy())
