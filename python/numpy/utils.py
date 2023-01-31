from typing import Optional, Union, Tuple

import numpy as np


def np_unique_int_fast(
    arr: np.ndarray,
    maxnum: Optional[int] = None,
    return_inverter: bool = False
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Fast implementation of np.unique(arr, return_index=True, return_inverse=True)

    Runs O(n) instead of O(n*log(n)) since np.unique sorts the input and this does not.

    Args:
        arr (dtype: np.int, shape: (n,)): Input array. This must be 1-D and 
        maxnum: The maximum allowed value in arr
        return_inverter: Also return an inverter that maps values in arr to indices of unique elements

    Returns:
        index (dtype: np.uint, shape: (n_unique,)): Indices of the last occurrences of the unique values in the original
            array. Note that for np.unique, this will return the indices of the first occurrences (not the last)
        inverse (dtype: np.uint, shape: (n,)): Index of unique for each element. This maps arr to the
            range [0, n_unique - 1]
        inverter (dtype: np.uint32, shape: (arr.max(),)): Optional. Only returned if return_inverter=True. Large array
            that maps values in arr to index of unique elements. This can be used to get the unique values for new
            input. Note that `inverter[arr] == inverse`

    Example:
        >>> np_unique_int_fast(np.array([3, 3, 2, 1]))
        (array([3, 2, 1], dtype=uint64), array([2, 2, 1, 0], dtype=uint32))
        >>> np_unique_int_fast(np.array([3, 3, 2, 1]), return_inverter=True)
        (array([3, 2, 1], dtype=uint64), array([2, 2, 1, 0], dtype=uint32), array([0, 0, 1, 2], dtype=uint32))
        >>> np_unique_int_fast(np.array([1, 1, 2, 3, 2, 1, 1]))
        (array([6, 4, 3], dtype=uint64), array([0, 0, 1, 2, 1, 0, 0], dtype=uint32))
    """
    if len(arr.shape) > 1:
        raise ValueError("Expected numpy array to be a vector")
    if arr.min() < 0:
        raise ValueError(f"arr values cannot be negative. Got arr.min()={arr.min()}")
    maxval = arr.max()
    if maxnum is None:
        maxnum = int(maxval + 1)  # Used as length of index array
    if maxval >= maxnum:
        raise ValueError(f"maxnum must be larger then arr.max(). Got maxnum={maxnum} != arr.max()={maxval}")
    exists = np.zeros(maxnum, dtype=bool)
    exists[arr] = True
    n_unique = exists.sum()
    index = np.empty(maxnum, dtype=np.uint64)
    index[arr] = np.arange(arr.shape[0])
    groups = np.zeros(maxnum, dtype=np.uint32)
    groups[exists] = np.arange(n_unique)
    if return_inverter:
        return index[exists], groups[arr], groups
    return index[exists], groups[arr]