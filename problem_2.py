from scipy.optimize import linprog
import numpy as np
from numpy import ndarray
from mat2d_round import *
from bicycle_iter import *

MaxBicycleNum = 1000  # 最大自行车投放数量

# 每小时的需求矩阵
DemandsHourly = [0,1,2,3,4,5,6,7,8,9,10,
                1,0,86,120,75,122,92,129,60,105,97,
                2,86,0,58,124,103,149,117,60,74,119,
                3,109,74,0,102,88,76,140,97,70,111,
                4,120,81,79,0,72,58,128,75,115,129,
                5,70,140,109,138,0,88,114,51,140,71,
                6,52,100,70,116,50,0,148,70,82,66,
                7,88,80,91,100,64,69,0,59,114,51,
                8,119,149,90,139,93,132,135,0,110,143,
                9,106,95,107,97,150,128,91,141,0,117,
                10,106,138,86,116,116,112,95,146,136,0]

DemandsHourly = np.array(DemandsHourly)
DemandsHourly: ndarray = DemandsHourly.reshape((11, 11))[1:, 1:]  # 去掉行头和列头

#10分钟需求矩阵
Demands10m: ndarray = (DemandsHourly/6).round(0).astype(int)
# 每小时的进车上限
BicycleIn = DemandsHourly.sum(0)
# 每小时的出车上限
BicycleOut = DemandsHourly.sum(1)

# 计算进车的比例

# 按行进行归一化的需求矩阵
NormalizedDemandsOut: ndarray = DemandsHourly / DemandsHourly.sum(1, keepdims=True)

# print(NormalizedDemandsOut.sum(1))


InitBicycleNumsList = []
max_ins: ndarray = Demands10m.sum(0)
max_outs: ndarray = Demands10m.sum(1)

# region 在调度周期末尾进行补齐
# bicycles = np.array([95, 103,  91, 110,  94,  98, 118,  98, 107, 100]) # 10分钟需求矩阵的平衡态
# # bicycles = np.array([301, 241, 352, 166, 383, 148, 118, 1188, 476, 616])
# out_gaps = np.zeros(10, dtype=int)
# cnt = 0

# while cnt < 100:
#     InitBicycleNums = bicycles.copy()
#     for epoch in range(0, 18):
#         bicycles, transfers, outs, ins = bicycle_epoch(bicycles, Demands10m)
#         # print(f"epoch {epoch+1}, out: {outs.sum().item()}, {outs}")

#     out_gaps = max_outs - outs
#     print(f"step {cnt:2d} out_gaps: {out_gaps} init {InitBicycleNums} outs {outs}")

#     InitBicycleNumsList.append(InitBicycleNums)
#     if out_gaps.sum().item() == 0 : break
#     bicycles = InitBicycleNums + out_gaps
#     cnt += 1
# endregion

# region 在第一次周转后进行补齐

# bicycles = np.array([95, 103,  91, 110,  94,  98, 118,  98, 107, 100]) # 10分钟需求矩阵的平衡态
bicycles: ndarray = max_outs.copy()
out_gaps = np.zeros(10, dtype=int)
cnt = 0

while cnt < 5:
    InitBicycleNums = bicycles.copy()
    total_paddings = np.zeros_like(max_outs)
    for epoch in range(0, 18):
        _, _, outs, _ = bicycle_epoch(bicycles, Demands10m) # 使用现有的自行车数量分布计算出车量
        padded_outs = np.maximum(outs, max_outs) # 为了保证每次出车都完全满足出车上限，进行补齐
        paddings = padded_outs - outs # 计算补齐量
        bicycles = bicycles + paddings # 进行补齐
        total_paddings = total_paddings + paddings
        bicycles, _, outs, _ = bicycle_epoch(bicycles, Demands10m) # 补齐后计算出车量
        # print(f"epoch {epoch+1}, out: {outs.sum().item()}, {outs}, paddings {paddings}")

    bicycles = InitBicycleNums + total_paddings
    InitBicycleNumsList.append(bicycles)
    print(f"step {cnt:2d} best_init {bicycles} padding {total_paddings}")

    # if out_gaps.sum().item() == 0 : break
    cnt += 1
# endregion

for InitBicycleNums in InitBicycleNumsList:
    print("Init Bicycle:\t", InitBicycleNums.sum(), InitBicycleNums)
    revenue = total_revenue(InitBicycleNums, Demands10m, mute=True)
    print(f"{BICYCLE_LIFE}年生命周期总利润: {revenue:.2f}元, 平均每辆车收益{revenue/InitBicycleNums.sum().item()}元")