from scipy.optimize import linprog
import numpy as np

MaxBicycleNum = 1000

Demands = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
           1, 0, 86, 120, 75, 122, 92, 129, 60, 105, 97,
           2, 86, 0, 58, 124, 103, 149, 117, 60, 74, 119,
           3, 109, 74, 0, 102, 88, 76, 140, 97, 70, 111,
           4, 120, 81, 79, 0, 72, 58, 128, 75, 115, 129,
           5, 70, 140, 109, 138, 0, 88, 114, 51, 140, 71,
           6, 52, 100, 70, 116, 50, 0, 148, 70, 82, 66,
           7, 88, 80, 91, 100, 64, 69, 0, 59, 114, 51,
           8, 119, 149, 90, 139, 93, 132, 135, 0, 110, 143,
           9, 106, 95, 107, 97, 150, 128, 91, 141, 0, 117,
           10, 106, 138, 86, 116, 116, 112, 95, 146, 136, 0]

Demands = np.array(Demands)
Demands = Demands.reshape((11, 11))[1:, 1:]
# 每小时的进车上限
BicycleIn = Demands.sum(0)
# 每小时的出车上限
BicycleOUT = Demands.sum(1)

# 计算进车的比例

# 按行进行归一化的需求矩阵
NormalizedDemandsOut = Demands / Demands.sum(1, keepdims=True)

# print(NormalizedDemandsOut.sum(1))

def ride_bicycle(bicycles : np.array, slice_num = 6):
    remains = bicycles - BicycleOUT
    remains = remains * (remains > 0).astype(int)
    print('remains: ', remains)

    out_bicycles = bicycles - remains
    total_out = out_bicycles.sum()
    out_bicycles = out_bicycles * NormalizedDemandsOut
    out_bicycles = out_bicycles.round(0).astype(int)
    # 检查四舍五入前后的数量是否一致
    r = total_out - out_bicycles.sum()
    if r > 0:
        gaps = Demands/slice_num - out_bicycles # 需求缺口量
        gaps = np.floor(gaps) * (gaps > 0).astype(int)
        gaps = gaps.flatten()
        adj = np.zeros_like(gaps, dtype=int)
        idx = gaps.argsort()
        for i in idx:
            if r <= 0: break
            if r >= gaps[i]:
                adj[i] += gaps[i]
                r = r - gaps[i]
            else:
                adj[i] += r
                r = 0
        adj = adj.reshape(out_bicycles.shape)
        print('r=', total_out - out_bicycles.sum(), ', sum adj: ', adj.sum())
        out_bicycles += adj
    elif r < 0:
        adj = np.zeros_like(out_bicycles, dtype=int)
        adj = adj.flatten()
        pass # todo

    print('out_bicycles: \n', out_bicycles, out_bicycles.sum())


InitBicycleNums = MaxBicycleNum / BicycleOUT.sum() * BicycleOUT
InitBicycleNums = InitBicycleNums.round(0).astype(int)
print("InitBicycleNums: ", InitBicycleNums*3)
ride_bicycle(InitBicycleNums)