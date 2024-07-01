import numpy as np

# 参数定义
C = 4000  # 每台单车成本
F = 500000  # 每年固定开支费用
R = 3  # 每笔订单收益
M = 0.15  # 日常维护费用比例
L = 5  # 单车使用年限
Rc = 0.1  # 报废车回收率

# 晴天天数比率
sunny_ratio = 10.19 / 15.39
rainy_ratio = 5.2 / 15.39

# 各区域需求数据（示例数据）
D = np.array([
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

# 预处理各区域单车需求总和
demand_sums = np.sum(D, axis=1)

# 初始化DP表格
max_bikes = 1000
years = 5
regions = 10

dp = np.zeros((years + 1, max_bikes + 1))

# 收益函数
def profit(region, bikes):
    orders = demand_sums[region] * bikes * sunny_ratio
    revenue = orders * R
    maintenance = revenue * M
    return revenue - maintenance - F / regions

# 动态规划求解
for t in range(1, years + 1):
    for x in range(max_bikes + 1):
        for i in range(regions):
            for k in range(x + 1):
                dp[t][x] = max(dp[t][x], dp[t - 1][x - k] + profit(i, k))

# 求解最优值
max_profit = np.max(dp[years])
print(f"最大盈利: {max_profit}")

# 求解最优投放策略
bikes_left = max_bikes
optimal_distribution = [0] * regions

for t in range(years, 0, -1):
    for i in range(regions):
        for k in range(bikes_left + 1):
            if dp[t][bikes_left] == dp[t - 1][bikes_left - k] + profit(i, k):
                optimal_distribution[i] += k
                bikes_left -= k
                break

print("最优投放策略:")
for region, bikes in enumerate(optimal_distribution):
    print(f"区域 {region + 1}: 投放 {bikes} 辆单车")
