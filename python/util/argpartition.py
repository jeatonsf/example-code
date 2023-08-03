from typing import List


def argpartition(nums: List[int], k: int) -> List[int]:
    nums = nums.copy()
    idxs = [i for i in range(len(nums))]
    lo, hi = 0, len(nums) - 1
    while lo < hi:
        p = partition(nums, idxs, lo, hi)
        if p < k:
            lo = p + 1
        elif p > k:
            hi = p - 1
        else:
            return idxs
    return idxs


def partition(nums: List[int], idxs: List[int], lo: int, hi: int) -> int:
    pivot = nums[hi]
    i = lo
    for j in range(lo, hi):
        if nums[j] < pivot:
            nums[i], nums[j] = nums[j], nums[i]
            idxs[i], idxs[j] = idxs[j], idxs[i]
            i += 1
    nums[i], nums[hi] = nums[hi], nums[i]
    idxs[i], idxs[hi] = idxs[hi], idxs[i]
    return i
