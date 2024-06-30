import numpy as np
from mat2d_round import mat2d_round

# 每小时的需求矩阵
DEMANDS_HOURLY = [0,1,2,3,4,5,6,7,8,9,10,
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
DEMANDS_HOURLY = (np.array(DEMANDS_HOURLY)/6).round(0).astype(int)
DEMANDS_HOURLY: np.ndarray
def bicycle_iter(bicycles: np.ndarray, demands: np.ndarray = DEMANDS_HOURLY):
    # 迭代一次自行车的输出与输入, bicycles为初始的自行车数量矩阵, demands 为各区域的需求矩阵
    # 所有参数都必须为整数
    max_bicycles_in = demands.sum(0)
    max_bicycles_out = demands.sum(1)

    remains = bicycles - max_bicycles_out # 出车以后剩余的自行车数
    remains = remains * (remains > 0).astype(int)

    out_bicycles = bicycles - remains
    total_out = out_bicycles.sum()
    norm_demands = demands / demands.sum(1, keepdims=True) # 按行对需求矩阵进行归一化，以此作为出车目的地的比率
    transfers = out_bicycles.reshape((-1, 1)) * norm_demands
    transfers = mat2d_round(transfers, dim=1) # 按行对转移矩阵进行四舍五入

    in_bicycles = transfers.sum(0) # 计算入车的数量
    assert np.all(demands >= transfers) and np.all(transfers >= 0), "发车量超出需求!"
    final_remains = remains + in_bicycles

    # print('bicycles:\t', bicycles, bicycles.sum())
    print("out_bicycles:\t", out_bicycles.sum(), out_bicycles)
    # print('remains:\t', remains, remains.sum())
    # print('in_bicycles:\t', in_bicycles, in_bicycles.sum())
    print("final_remains:\t", final_remains.sum(), final_remains)

    return final_remains, transfers
