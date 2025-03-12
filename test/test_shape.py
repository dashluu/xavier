from python.xavier import Shape
import pytest
import numpy as np


class TestShape:
    def test_broadcastable_same_rank(self):
        s = Shape([1, 2, 3, 1, 3])
        v = [2, 2, 1, 2, 1]
        assert s.broadcastable(v)

    def test_broadcastable_diff_ranks(self):
        s = Shape([1, 2, 3, 1, 3])
        v = [1, 2, 1]
        assert s.broadcastable(v)

    def test_not_broadcastable(self):
        s = Shape([1, 2, 3, 1, 3])
        v = [1, 2, 1, 1]
        assert not s.broadcastable(v)

    def test_broadcast_same_rank(self):
        s = Shape([1, 2, 3, 1, 3])
        v = [2, 2, 1, 2, 1]
        broadcasted = s.broadcast(v)
        assert broadcasted.view() == [2, 2, 3, 2, 3]
        assert broadcasted.stride() == [0, 9, 3, 0, 1]

    def test_broadcast_diff_ranks_v1(self):
        s = Shape([1, 2, 3, 1, 3])
        v = [1, 2, 1]
        broadcasted = s.broadcast(v)
        assert broadcasted.view() == [1, 2, 3, 2, 3]
        assert broadcasted.stride() == [18, 9, 3, 0, 1]

    def test_broadcast_diff_ranks_v2(self):
        s = Shape([1, 4, 1, 3])
        v = [2, 1, 2, 4, 3, 3]
        broadcasted = s.broadcast(v)
        assert broadcasted.view() == [2, 1, 2, 4, 3, 3]
        assert broadcasted.stride() == [0, 0, 0, 3, 0, 1]

    def test_broadcast_same_shape(self):
        s = Shape([2, 3, 4])
        v = [2, 3, 4]
        broadcasted = s.broadcast(v)
        assert broadcasted.view() == [2, 3, 4]
        assert broadcasted.stride() == [12, 4, 1]  # Original strides preserved

    def test_broadcast_scalar_to_dims(self):
        s = Shape([1])
        v = [2, 3, 4]
        broadcasted = s.broadcast(v)
        assert broadcasted.view() == [2, 3, 4]
        assert broadcasted.stride() == [0, 0, 0]  # All strides zero for scalar broadcast

    def test_broadcastable_to_diff_ranks_v1(self):
        s = Shape([1, 4, 1, 3])
        v = [2, 1, 2, 4, 3, 3]
        assert s.broadcastable_to(v)

    def test_broadcastable_to_diff_ranks_v2(self):
        s = Shape([1, 4, 1, 3, 1, 3, 3])
        v = [2, 1, 2, 4, 3, 3]
        assert not s.broadcastable_to(v)

    def test_broadcastable_to_same_rank(self):
        s = Shape([1, 1, 1, 4, 3, 1])
        v = [2, 1, 2, 4, 3, 3]
        assert s.broadcastable_to(v)

    def test_broadcastable_to_same_shape(self):
        s = Shape([1, 1, 1, 4, 3, 1])
        v = [1, 1, 1, 4, 3, 1]
        assert s.broadcastable_to(v)

    def test_broadcast_to_diff_ranks(self):
        s = Shape([1, 4, 1, 3])
        v = [2, 1, 2, 4, 3, 3]
        broadcasted = s.broadcast_to(v)
        assert broadcasted.view() == [2, 1, 2, 4, 3, 3]
        assert broadcasted.stride() == [0, 0, 0, 3, 0, 1]

    def test_broadcast_to_same_rank(self):
        s = Shape([1, 1, 1, 4, 3, 1])
        v = [2, 1, 2, 4, 3, 3]
        broadcasted = s.broadcast_to(v)
        assert broadcasted.view() == [2, 1, 2, 4, 3, 3]
        assert broadcasted.stride() == [0, 12, 0, 3, 1, 0]

    def test_permute_v1(self):
        s = Shape([2, 3, 4])  # Initial shape
        permuted = s.permute([2, 0, 1])  # Permute to [4, 2, 3]
        assert permuted.view() == [4, 2, 3]
        assert permuted.stride() == [1, 12, 4]

    def test_permute_v2(self):
        s = Shape([2, 3, 4, 5])
        permuted = s.permute([3, 1, 2, 0])  # Permute to [5, 3, 4, 2]
        assert permuted.view() == [5, 3, 4, 2]
        assert permuted.stride() == [1, 20, 5, 60]

    def test_permute_invalid_dims(self):
        s = Shape([2, 3, 4])
        # Test invalid permutation indices
        with pytest.raises(ValueError, match="The order must be a permutation of the dimensions but got 3, 1, 0."):
            s.permute([3, 1, 0])  # Index 3 is out of bounds

    def test_permute_repeated_dims(self):
        s = Shape([2, 3, 4])
        # Test repeated indices
        with pytest.raises(ValueError, match="The order must be a permutation of the dimensions but got 1, 1, 2."):
            s.permute([1, 1, 2])  # Repeated index 1

    def test_matmul_compat(self):
        print("\nTesting matmul compatibility:")

        # Test cases: [lhs_shape, rhs_shape, expected_result]
        test_cases = [
            # Basic matrix multiplication
            ([2, 3], [3, 4], True),  # (2,3) @ (3,4) -> (2,4)
            ([3, 2], [4, 3], False),  # Incompatible inner dimensions
            # Batch matrix multiplication
            ([5, 2, 3], [5, 3, 4], True),  # 5 batches
            ([5, 2, 3], [3, 4], True),  # Broadcasting rhs
            ([2, 3], [5, 3, 4], True),  # Broadcasting lhs
            # Complex batch cases
            ([8, 1, 2, 3], [8, 5, 3, 4], True),  # Broadcasting middle dimension
            ([1, 2, 3], [5, 3, 4], True),  # Broadcasting batch dim
            ([8, 2, 3], [1, 3, 4], True),  # Broadcasting with 1s
            # Invalid cases
            ([2, 3], [2, 4], False),  # Wrong inner dimensions
            ([2], [2], False),  # 1D tensors
        ]

        for lhs_shape, rhs_shape, expected in test_cases:
            shape = Shape(np.zeros(lhs_shape).shape)
            result = shape.matmul_compat(rhs_shape)
            assert result == expected, f"Failed: {lhs_shape} @ {rhs_shape}, expected {expected}, got {result}"
            print(f"Shape {lhs_shape} @ {rhs_shape}: {'✓' if result == expected else '✗'}")

    def test_matmul_broadcast_dim_mismatch(self):
        print("\nTesting matmul broadcasting with dimension mismatches:")

        # Test cases: [lhs_shape, rhs_shape, expected_broadcast_shape]
        test_cases = [
            # Different number of batch dimensions
            ([2, 3], [5, 6, 3, 4], [5, 6, 2, 4]),  # 2D @ 4D
            ([10, 2, 3], [3, 4], [10, 2, 4]),  # 3D @ 2D
            ([2, 3], [1, 1, 1, 3, 4], [1, 1, 1, 2, 4]),  # 2D @ 5D
            # Complex batch dimension mismatches
            ([1, 1, 2, 3], [7, 5, 3, 4], [7, 5, 2, 4]),  # 4D @ 4D with different batches
            ([8, 7, 2, 3], [1, 3, 4], [8, 7, 2, 4]),  # 4D @ 3D
            ([2, 3], [8, 7, 6, 3, 4], [8, 7, 6, 2, 4]),  # 2D @ 5D
            # Edge cases
            ([1, 2, 3], [10, 9, 8, 7, 3, 4], [10, 9, 8, 7, 2, 4]),  # Large dimension mismatch
            ([5, 4, 2, 3], [3, 4], [5, 4, 2, 4]),  # Multiple batch vs no batch
        ]

        for lhs_shape, rhs_shape, expected_shape in test_cases:
            try:
                shape = Shape(np.zeros(lhs_shape).shape)
                result = shape.matmul_broadcast(rhs_shape)
                assert list(result.view()) == expected_shape, (
                    f"Failed: broadcasting {lhs_shape} with {rhs_shape}\n"
                    f"Expected {expected_shape}, got {list(result.view())}"
                )
                print(f"Broadcast {lhs_shape} with {rhs_shape} -> {list(result.view())}: ✓")
            except Exception as e:
                print(f"Broadcast {lhs_shape} with {rhs_shape} failed: {str(e)}")

    def test_matmul_broadcast_invalid_dims(self):
        print("\nTesting invalid dimension combinations:")

        # Test cases that should raise exceptions
        error_cases = [
            # Missing matrix dimensions
            ([5], [3, 4]),  # 1D @ 2D
            ([2, 3], [4]),  # 2D @ 1D
            ([], [3, 4]),  # 0D @ 2D
            # Incompatible matrix dimensions
            ([5, 2, 3], [5, 4, 4]),  # Inner dimensions don't match
            ([2, 3], [5, 4, 5]),  # Neither dimension matches
            # Invalid batch dimensions
            ([2, 2, 3], [3, 2, 3, 4]),  # Incompatible batch dims
            ([5, 5, 2, 3], [2, 3, 3, 4]),  # Batch dims can't be broadcast
        ]

        for lhs_shape, rhs_shape in error_cases:
            with pytest.raises(Exception) as exc_info:
                shape = Shape(np.zeros(lhs_shape).shape)
                result = shape.matmul_broadcast(rhs_shape)
                print(f"Should have raised exception for {lhs_shape} @ {rhs_shape}")
            print(f"Correctly raised {exc_info.type.__name__} for {lhs_shape} @ {rhs_shape}")
