from __future__ import annotations
from numbers import Number
from typing import Sequence

import numpy as np
from scipy.optimize import fminbound
from scipy.stats import norm


class GaussianKernelSmoother:
    def __init__(self, k: Optional[int] = None, scale: Optional[float] = None) -> None:
        """Non-parametric 1-D regressor using kernel smoothing with Gaussian kernel

        Predicted targets are a gaussian weighting of the training data centered around the input
        feature value. The Gaussian is parameterized by `scale`. Kernel smoothing does not depend
        on knowing the distribution of input data but it does assume the following
          1. The train data is similarly distrubuted to the test data
          2. The true target function is continuous
        See wikipedia for details
        https://en.wikipedia.org/wiki/Kernel_smoother#Gaussian_kernel_smoother

        Args:
            k: Number of nearest neighbors to use in prediction. Lower values of k decrease compute
                time. If None, use all samples
            scale: scale parameter of Gaussian kernel. If None, this will be learned in fit by
                minimizing the leave one out residual sum of squares

        Example:
            >>> GaussianKernelSmoother().fit_predict([1, 2, 3, 4, 5], [2.0, 4, 6, 8, 10])
            array([3.04017959, 4.2576853, 6.0, 7.7423147, 8.95982041])
        """
        self.k_ = k
        self.scale = scale

    def fit(self, x: Sequence[Number], y: Sequence[Number]) -> GaussianKernelSmoother:
        x, y = np.array(x), np.array(y)
        self.x, self.y = np.array(x), np.array(y)
        idx = np.argsort(self.x)
        self.x, self.y = self.x[idx], self.y[idx]
        self._check_fit()
        
        def _rss(scale: float) -> float:
            """Residual sum of squares"""
            return np.sum(np.square(y - self._smooth_loo(x, y, np.abs(scale))))
        
        diffs = np.abs(self.x[:-1] - self.x[1:])
        if self.scale is None:
            self.scale = np.abs(fminbound(_rss, np.min(diffs), np.std(x), disp=False))
        return self

    def fit_predict(self, x: Sequence[Number], y: Sequence[Number]) -> np.ndarray:
        return self.fit(x, y).predict(x)

    @property
    def k(self) -> int:
        if self.k_ is None:
            return self.x.shape[0] - 1
        return self.k_

    def predict(self, x: Sequence[Number]) -> np.ndarray:
        x = np.array(x)
        y_smooth = np.empty((len(x),), dtype=self.y.dtype)
        for i in range(len(x)):
            diffs = np.abs(self.x - x[i])
            idx = np.argpartition(diffs, self.k)[:self.k + 1]
            w = norm.pdf(self.x[idx] - x[i], loc=0, scale=self.scale)
            y_smooth[i] = np.dot(self.y[idx], w) / np.sum(w)
        return y_smooth

    def _check_fit(self) -> None:
        if len(self.x.shape) != 1:
            raise ValueError(f"Can only smooth a 1D curve. Got shape: {self.x.shape}")
        if self.x.shape != self.y.shape:
            raise ValueError("x and y must have the same shape")
        if self.k >= self.x.shape[0]:
            raise ValueError(f"k={self.k} should be less than len(x)={len(self.x)}")
    
    def _smooth_loo(self, x: np.ndarray, y: np.ndarray, scale: float) -> np.ndarray:
        """Smooth leave one out"""
        y_smooth = np.empty_like(y)
        for i in range(len(x)):
            idx = np.argpartition(np.abs(x - x[i]), self.k)[:self.k + 1]
            idx = np.delete(idx, np.argwhere(idx == i))
            w = norm.pdf(x[idx] - x[i], loc=0, scale=scale)
            y_smooth[i] = np.dot(y[idx], w) / np.sum(w)
        return y_smooth