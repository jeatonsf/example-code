import unittest

import numpy as np

from gaussian_kernel_smoother import GaussianKernelSmoother


class TestNpUniqueIntFast(unittest.TestCase):
    def test_docstring_example(self) -> None:
        np.random.seed(0)
        x = np.random.uniform(-3, 3, size=512)
        y = x ** 2 + np.random.normal(0, 1, size=len(x))
        smoother = GaussianKernelSmoother().fit(x, y)
        y_pred = smoother.predict(x)
        assert all(y_pred != y), "y_pred should never exactly equal y"
        assert smoother.predict([0])[0] < smoother.predict([-1])[0], "middle should be lower than left"
        assert smoother.predict([0])[0] < smoother.predict([1])[0], "middle should be lower than right"
        diff = np.abs((x ** 2) - smoother.predict(x))
        assert np.mean(diff) < 0.5, "should be reasonably close to true function y=x**2"
        assert np.std(diff) < 0.5, "should be reasonably close to true function y=x**2"
