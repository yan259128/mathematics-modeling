import numpy as np
from scipy.optimize import linprog

# 输入数据
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

# 参数
cost_per_bike = 4000
fixed_cost_per_year = 500000
revenue_per_order = 3
maintenance_cost_rate = 0.165  # 取15%-18%的中间值
bike_lifetime = 5
recovery_rate = 0.1
sunny_hours_per_day = 12
rainy_hours_per_day = 0

# 计算晴天和雨天数量
sunny_days_ratio = 5.2
rainy_days_ratio = 10.19
total_days = 365 * 5
sunny_days = (sunny_days_ratio / (sunny_days_ratio + rainy_days_ratio)) * total_days
rainy_days = (rainy_days_ratio / (sunny_days_ratio + rainy_days_ratio)) * total_days

# 计算各区域年收入
hourly_demand = np.sum(demand_matrix, axis=1)
print("hourly_demand", hourly_demand)
annual_revenue_per_bike = hourly_demand * revenue_per_order * sunny_hours_per_day * sunny_days
print("annual_revenue_per_bike", annual_revenue_per_bike)

# 目标函数系数 (收入 - 维护成本)
c = -annual_revenue_per_bike * (1 - maintenance_cost_rate)

# 约束条件
A_eq = np.ones((1, 10))
b_eq = [1000]
# 设置每个区域投放单车的上下限，允许为0
bounds = [(0, None) for _ in range(10)]

# 求解问题1：优化投放方案
result1 = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

# 输出结果
if result1.success:
    optimal_distribution1 = result1.x
    total_revenue1 = np.sum(annual_revenue_per_bike * optimal_distribution1)
    total_cost1 = 1000 * cost_per_bike + fixed_cost_per_year * bike_lifetime + total_revenue1 * maintenance_cost_rate
    total_profit1 = total_revenue1 - total_cost1
    print("问题1 最优投放方案：", optimal_distribution1)
    print("问题1 五年总利润：", total_profit1)
else:
    print("问题1 求解失败")

# import numpy as np
# from simanneal import Annealer
#
# # 初始化参数
# cost_per_bike = 4000
# fixed_annual_cost = 500000
# revenue_per_order = 3
# maintenance_cost_percentage = 0.165  # 平均值
# bike_lifetime = 5  # 年
# scrap_value_percentage = 0.10
# initial_bikes = 1000
#
# # 区域需求矩阵 (从文档中提取的示例数据)
# demand_matrix = np.array([
#     [0, 86, 120, 75, 122, 92, 129, 60, 105, 97],
#     [86, 0, 58, 124, 103, 149, 117, 60, 74, 119],
#     [109, 74, 0, 102, 88, 76, 140, 97, 70, 111],
#     [120, 81, 79, 0, 72, 58, 128, 75, 115, 129],
#     [70, 140, 109, 138, 0, 88, 114, 51, 140, 71],
#     [52, 100, 70, 116, 50, 0, 148, 70, 82, 66],
#     [88, 80, 91, 100, 64, 69, 0, 59, 114, 51],
#     [119, 149, 90, 139, 93, 132, 135, 0, 110, 143],
#     [106, 95, 107, 97, 150, 128, 91, 141, 0, 117],
#     [106, 138, 86, 116, 116, 112, 95, 146, 136, 0]
# ])
#
#
# class BikeAllocationProblem(Annealer):
#
#     def __init__(self, state):
#         super(BikeAllocationProblem, self).__init__(state)
#
#     def move(self):
#         i = np.random.randint(0, 10)
#         j = np.random.randint(0, 10)
#
#         if i != j and self.state[i] > 0:
#             change = np.random.randint(1, min(self.state[i], 100) + 1)
#             self.state[i] -= change
#             self.state[j] += change
#
#     def energy(self):
#         total_bikes = sum(self.state)
#         total_revenue = 0
#
#         for i in range(10):
#             for j in range(10):
#                 total_revenue += demand_matrix[i, j] * revenue_per_order * self.state[i] * 3
#
#         annual_revenue = total_revenue * (10.19 / 15.39)  # 计算年收入，考虑晴天和雨天比例
#         total_annual_cost = total_bikes * cost_per_bike + fixed_annual_cost + total_revenue * maintenance_cost_percentage
#         scrap_value = total_bikes * cost_per_bike * scrap_value_percentage
#
#         five_year_profit = 5 * (annual_revenue - total_annual_cost) + scrap_value
#         return -five_year_profit  # 因为Annealer是最小化能量，我们需要最大化利润，所以取负值
#
#
# # 初始化参数
# initial_allocation = [100] * 10  # 初始投放方案，每个区域100台
#
# # 创建并运行模拟退火算法
# bike_problem = BikeAllocationProblem(initial_allocation)
# bike_problem.Tmax = 25000  # 初始温度
# bike_problem.Tmin = 0.1  # 最小温度
# bike_problem.steps = 50000  # 总步数
# bike_problem.updates = 100  # 更新步数
#
# optimal_allocation, optimal_profit = bike_problem.anneal()
#
# # 打印结果
# print("最佳投放方案:", optimal_allocation)
# print("五年后的最大盈利:", -optimal_profit)



# import numpy as np
#
# # 参数定义
# C = 4000  # 每台单车成本
# F = 500000  # 每年固定开支费用
# R = 3  # 每笔订单收益
# M = 0.15  # 日常维护费用比例
# L = 5  # 单车使用年限
# Rc = 0.1  # 报废车回收率
#
# # 晴天天数比率
# sunny_ratio = 10.19 / 15.39
# rainy_ratio = 5.2 / 15.39
#
# # 各区域需求数据（示例数据）
# D = np.array([
#     [0, 86, 120, 75, 122, 92, 129, 60, 105, 97],
#     [86, 0, 58, 124, 103, 149, 117, 60, 74, 119],
#     [109, 74, 0, 102, 88, 76, 140, 97, 70, 111],
#     [120, 81, 79, 0, 72, 58, 128, 75, 115, 129],
#     [70, 140, 109, 138, 0, 88, 114, 51, 140, 71],
#     [52, 100, 70, 116, 50, 0, 148, 70, 82, 66],
#     [88, 80, 91, 100, 64, 69, 0, 59, 114, 51],
#     [119, 149, 90, 139, 93, 132, 135, 0, 110, 143],
#     [106, 95, 107, 97, 150, 128, 91, 141, 0, 117],
#     [106, 138, 86, 116, 116, 112, 95, 146, 136, 0]
# ])
#
# # 预处理各区域单车需求总和
# demand_sums = np.sum(D, axis=1)
#
# # 初始化DP表格
# max_bikes = 1000
# years = 5
# regions = 10
#
# dp = np.zeros((years + 1, max_bikes + 1))
#
# # 收益函数
# def profit(region, bikes):
#     orders = demand_sums[region] * bikes * sunny_ratio
#     revenue = orders * R
#     maintenance = revenue * M
#     return revenue - maintenance - F / regions
#
# # 动态规划求解
# for t in range(1, years + 1):
#     for x in range(max_bikes + 1):
#         for i in range(regions):
#             for k in range(x + 1):
#                 dp[t][x] = max(dp[t][x], dp[t - 1][x - k] + profit(i, k))
#
# # 求解最优值
# max_profit = np.max(dp[years])
# print(f"最大盈利: {max_profit}")
#
# # 求解最优投放策略
# bikes_left = max_bikes
# optimal_distribution = [0] * regions
#
# for t in range(years, 0, -1):
#     for i in range(regions):
#         for k in range(bikes_left + 1):
#             if dp[t][bikes_left] == dp[t - 1][bikes_left - k] + profit(i, k):
#                 optimal_distribution[i] += k
#                 bikes_left -= k
#                 break
#
# print("最优投放策略:")
# for region, bikes in enumerate(optimal_distribution):
#     print(f"区域 {region + 1}: 投放 {bikes} 辆单车")
