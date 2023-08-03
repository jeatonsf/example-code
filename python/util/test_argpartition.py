import unittest

import numpy as np

from argpartition import argpartition


class TestArgpartition(unittest.TestCase):
    def test_argpartition(self) -> None:
        np.random.seed(0)
        nums = list(np.random.randint(0, 255, 2048))
        k = 128
        idx = argpartition(nums, k)
        for i in idx[:k]:
            assert nums[i] <= nums[idx[k]]
        for i in idx[k:]:
            assert nums[i] >= nums[idx[k]]
