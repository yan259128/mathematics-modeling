import numpy as np

# 定义常量
N = 10  # 区域数量
B = 1000  # 总投入单车数量
C = 4000  # 每台单车成本
F = 500000  # 每年固定开支
R = 3  # 每笔订单收益
M = 0.165  # 日常维护费用占比，取中间值
L = 5  # 单车使用年限
P = 10.19 / (10.19 + 5.2)  # 晴天比例
D = 5.2 / (10.19 + 5.2)  # 雨天比例

# 需求矩阵（示例数据）
demand_matrix = np.array([
    [0, 86, 120, 75, 122, 92, 129, 60, 105, 97],
    [86, 0, 58, 124, 103, 149, 117, 60, 74, 119],
    [109, 74, 0, 102, 88, 76, 140, 97, 70, 111],
    [120, 81, 79, 0, 72, 58, 128, 75, 115, 129],
    [70, 140, 109, 138, 0, 88, 114, 51, 140, 71],
    [52, 100, 70, 116, 50, 0, 148, 70, 82, 66],
    [88, 80, 91, 100, 64, 69, 0, 59, 114, 51],
    [119, 149, 90, 139, 93, 132, 135, 0, 110, 143],
    [106, 95, 107, 97, 150, 128, 91, 141, 0, 117],
    [106, 138, 86, 116, 116, 112, 95, 146, 136, 0]
])

# 初始化dp数组，dp[i][j]表示在前i个区域投放j台单车时的最大盈利
dp = np.zeros((N + 1, B + 1))

# 初始化分配矩阵，allocation[i][j]表示在前i个区域投放j台单车时，第i个区域的分配数量
allocation = np.zeros((N + 1, B + 1), dtype=int)

# 计算晴天和雨天的单车使用时长
sunny_hours = 12
rainy_hours = 0

# 动态规划求解
for i in range(1, N + 1):
    hourly_demand = demand_matrix[i - 1]
    total_demand_sunny = hourly_demand * sunny_hours * P
    total_demand_rainy = hourly_demand * rainy_hours * D
    total_demand = total_demand_sunny + total_demand_rainy
    income_per_bike = np.sum(total_demand) * R * 365 * L  # 每台单车的总收入
    maintenance_cost_per_bike = income_per_bike * M  # 每台单车的维护成本

    for j in range(B + 1):
        for k in range(j + 1):  # k表示在第i个区域投放的单车数量
            cost = k * C + F * L + k * maintenance_cost_per_bike
            profit = k * income_per_bike - cost
            if dp[i][j] < dp[i - 1][j - k] + profit:
                dp[i][j] = dp[i - 1][j - k] + profit
                allocation[i][j] = k

# 回溯找到最佳投放方案
best_allocation = np.zeros(N, dtype=int)
remaining_bikes = B
for i in range(N, 0, -1):
    best_allocation[i - 1] = allocation[i][remaining_bikes]
    remaining_bikes -= allocation[i][remaining_bikes]

max_profit = dp[N][B]

print("最大盈利:", max_profit)
print("最佳投放方案:", best_allocation)
