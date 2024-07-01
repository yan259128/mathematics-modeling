import numpy as np
import random


def mat2d_round(x: np.ndarray, dim: int = 1, eps=1e-6) -> np.ndarray:
    # 对矩阵按行或列进行四舍五入, 并保证结果的和等于原矩阵的和的四舍五入结果
    assert dim <= 1 and len(x.shape) == 2, "输入参数的维度"
    total_sum = x.sum().round(0).astype(int)
    sum = x.sum(dim, keepdims=True)
    sum = sum.round(0).astype(int)

    round_x = np.floor(x).astype(int)
    num_to_fix = sum - round_x.sum(dim, keepdims=True)
    if total_sum != sum.sum():
        pos = np.unravel_index(num_to_fix.argmax(), num_to_fix.shape)
        num_to_fix[pos] += total_sum - sum.sum()

    diff = x - round_x
    # 避免完全一样的diff
    diff += eps * np.random.rand(*diff.shape)

    idx = (-diff).argsort(axis=dim)
    idx = np.take_along_axis(idx, num_to_fix - 1, axis=dim)
    threshold = np.take_along_axis(diff, idx, axis=dim)
    mask = (diff >= threshold).astype(int)
    mask *= (num_to_fix != 0).astype(int)
    ret = round_x + mask
    assert total_sum == ret.sum(), "四舍五入失败"
    return ret