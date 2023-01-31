import unittest

import numpy as np

from fast_unique import np_unique_int_fast


class TestNpUniqueIntFast(unittest.TestCase):
    def test_docstring_example(self) -> None:
        index, inverse = np_unique_int_fast(np.array([3, 3, 2, 1]))
        self.assertListEqual(list(index), [3, 2, 1])
        self.assertListEqual(list(inverse), [2, 2, 1, 0])
        
        index, inverse, inverter = np_unique_int_fast(np.array([3, 3, 2, 1]), return_inverter=True)
        self.assertListEqual(list(index), [3, 2, 1])
        self.assertListEqual(list(inverse), [2, 2, 1, 0])
        self.assertListEqual(list(inverter), [0, 0, 1, 2])

        index, inverse = np_unique_int_fast(np.array([1, 1, 2, 3, 2, 1, 1]))
        self.assertListEqual(list(index), [6, 4, 3])
        self.assertListEqual(list(inverse), [0, 0, 1, 2, 1, 0, 0])

    def test_match_np_unique(self) -> None:
        np.random.seed(0)
        nums = np.random.randint(0, 7, size=256)
        _, expect_indices, expect_inverse = np.unique(nums, return_index=True, return_inverse=True)
        actual_indices, actual_inverse = np_unique_int_fast(nums)
        self.assertListEqual(list(nums[actual_indices]), list(nums[expect_indices]))
        self.assertListEqual(list(actual_inverse), list(expect_inverse))
