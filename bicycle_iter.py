import numpy as np
import math
from mat2d_round import mat2d_round

# 每小时的需求矩阵
DEMANDS_HOURLY: np.ndarray = np.array(
                [0,1,2,3,4,5,6,7,8,9,10,
                1,0,86,120,75,122,92,129,60,105,97,
                2,86,0,58,124,103,149,117,60,74,119,
                3,109,74,0,102,88,76,140,97,70,111,
                4,120,81,79,0,72,58,128,75,115,129,
                5,70,140,109,138,0,88,114,51,140,71,
                6,52,100,70,116,50,0,148,70,82,66,
                7,88,80,91,100,64,69,0,59,114,51,
                8,119,149,90,139,93,132,135,0,110,143,
                9,106,95,107,97,150,128,91,141,0,117,
                10,106,138,86,116,116,112,95,146,136,0]).reshape(11,11)[1:,1:]
DEMANDS_10M: np.ndarray = (np.array(DEMANDS_HOURLY) / 6).round(0).astype(int)

BICYCLE_FIX_COST = 4000  # 每辆自行车成本/元
FIX_COST_ANNUAL = 500000  # 每年固定成本/元
BICYCLE_LIFE = 5  # 自行车使用年限
BICYCLE_RECOVERY_RATE = 0.1  # 自行车报废回收率
BICYCLE_MAINT_RATIO_DAILY = 0.15  # 每日维护费用比例: 15%~18%
TRANSFER_REVENUE = 3  # 每笔订单收益/元
OP_HOURS_FINE_DAY = 12  # 晴天运营小时数
OP_HOURS_RAINY_DAY = 0  # 雨天运营小时数
FINE_DAY_RATIO = 10.19 / (5.2 + 10.19)  # 晴天比例
RAINY_DAY_RATIO = 1 - FINE_DAY_RATIO  # 晴天比例
DISPATCH_INTERVAL = 3  # 调度间隔小时数
TRANSFER_TIME = 1 / 6  # 自行车单次使用小时数


def bicycle_epoch(
    bicycles: np.ndarray,
    demands: np.ndarray = DEMANDS_HOURLY,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # 迭代一次自行车的输出与输入, bicycles为初始的自行车数量矩阵, demands 为各区域的需求矩阵
    # 所有参数都必须为整数
    max_bicycles_in = demands.sum(0)
    max_bicycles_out = demands.sum(1)

    remains = bicycles - max_bicycles_out  # 出车以后剩余的自行车数
    remains = remains * (remains > 0).astype(int)

    out_bicycles = bicycles - remains
    total_out = out_bicycles.sum()
    norm_demands = demands / demands.sum(
        1, keepdims=True
    )  # 按行对需求矩阵进行归一化，以此作为出车目的地的比率
    transfers = out_bicycles.reshape((-1, 1)) * norm_demands
    transfers = mat2d_round(transfers, dim=1)  # 按行对转移矩阵进行四舍五入

    in_bicycles = transfers.sum(0)  # 计算入车的数量
    assert np.all(demands >= transfers) and np.all(transfers >= 0), "发车量超出需求!"
    final_remains = remains + in_bicycles

    # print('bicycles:\t', bicycles, bicycles.sum())
    # print("out_bicycles:\t", out_bicycles.sum(), out_bicycles)
    # print('remains:\t', remains, remains.sum())
    # print('in_bicycles:\t', in_bicycles, in_bicycles.sum())
    # print("final_remains:\t", final_remains.sum(), final_remains)

    return final_remains, transfers, out_bicycles, in_bicycles


def total_revenue(
    bicycles: np.ndarray,
    demands: np.ndarray = DEMANDS_HOURLY,
    demands_interval=TRANSFER_TIME, 
    iter_interval=DISPATCH_INTERVAL,
    total_period_in_year = BICYCLE_LIFE,
    mute: bool=False,
)-> float:
    total_bicycle = int(bicycles.sum())

    life_cycle_iters = round(total_period_in_year * 365 * FINE_DAY_RATIO) * (
        OP_HOURS_FINE_DAY // iter_interval
    )
    life_cycle_iters += round(total_period_in_year * 365 * RAINY_DAY_RATIO) * (
        OP_HOURS_RAINY_DAY // iter_interval
    )

    epoches = math.floor(iter_interval // demands_interval)
    iter_transfer = 0
    for i in range(epoches):
        bicycles, _, outs, _ = bicycle_epoch(bicycles, demands)
        if not mute: print(f"epoch {i+1}, out: {int(outs.sum())}, {outs}")
        iter_transfer += outs.sum()
    iter_income = iter_transfer * TRANSFER_REVENUE
    iter_income *= 1 - BICYCLE_MAINT_RATIO_DAILY
    total_income = iter_income * life_cycle_iters

    bicycle_cost = BICYCLE_FIX_COST * total_bicycle
    bicycle_cost *= 1 - BICYCLE_RECOVERY_RATE
    total_cost = bicycle_cost + FIX_COST_ANNUAL * total_period_in_year
    return float(total_income - total_cost)
