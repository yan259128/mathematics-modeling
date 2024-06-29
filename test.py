from scipy.optimize import linprog
import numpy as np
from mat2d_round import mat2d_round
from bicycle_iter import bicycle_iter

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
DemandsHourly = DemandsHourly.reshape((11, 11))[1:, 1:]  # 去掉行头和列头
# 每小时的进车上限
BicycleIn = DemandsHourly.sum(0)
# 每小时的出车上限
BicycleOut = DemandsHourly.sum(1)

# 计算进车的比例

# 按行进行归一化的需求矩阵
NormalizedDemandsOut = DemandsHourly / DemandsHourly.sum(1, keepdims=True)

# print(NormalizedDemandsOut.sum(1))


# InitBicycleNums = MaxBicycleNum / BicycleOut.sum() * BicycleOut
# InitBicycleNums = InitBicycleNums.round(0).astype(int)

# InitBicycleNums = np.zeros(10, dtype=int)
# InitBicycleNums[7] = 5000
# InitBicycleNums[8] = 5000
# InitBicycleNums[9] = 5000

InitBicycleNums = np.array([95, 103,  91, 110,  94,  98, 118,  98, 107, 100])
# InitBicycleNums = np.array([126, 134, 118, 143, 124, 127, 120, 113, 173, 133])
print("Init Bicycle:\t", InitBicycleNums.sum(), InitBicycleNums)
# print("BicycleOut:\t", BicycleOut)
# print("BicycleIn:\t", BicycleIn)
for epoch in range(0, 18 * 10):
    print(f"Epoch {epoch} start")
    InitBicycleNums, _ = bicycle_iter(InitBicycleNums, (DemandsHourly/6).round(0).astype(int))
