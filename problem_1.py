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
