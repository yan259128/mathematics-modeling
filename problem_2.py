import numpy as np
from scipy.optimize import minimize

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
annual_revenue_per_bike = hourly_demand * revenue_per_order * sunny_hours_per_day * sunny_days

# 目标函数
def objective_function(x):
    bikes_distribution = x
    total_bikes = np.sum(bikes_distribution)
    total_revenue = np.sum(annual_revenue_per_bike * bikes_distribution)
    total_cost = total_bikes * cost_per_bike + fixed_cost_per_year * bike_lifetime + total_revenue * maintenance_cost_rate
    profit = total_revenue - total_cost
    return -profit  # 最大化利润等价于最小化负利润

# 初始猜测
x0 = np.ones(10) * 100000

# 边界条件
bounds = [(0, None) for _ in range(10)]

# 约束条件
constraints = [{'type': 'ineq', 'fun': lambda x: x}]  # 保证投放单车数不为负

# 求解问题2
result2 = minimize(objective_function, x0, bounds=bounds, constraints=constraints, method='SLSQP')

# 输出结果
if result2.success:
    optimal_distribution2 = result2.x
    total_bikes = np.sum(optimal_distribution2)
    total_revenue2 = np.sum(annual_revenue_per_bike * optimal_distribution2)
    total_cost2 = total_bikes * cost_per_bike + fixed_cost_per_year * bike_lifetime + total_revenue2 * maintenance_cost_rate
    max_profit = total_revenue2 - total_cost2
    print("问题2 最优单车总数：", total_bikes)
    print("问题2 最优投放方案：", optimal_distribution2)
    print("问题2 最大利润：", max_profit)
else:
    print("问题2 求解失败")
    print("状态信息：", result2.message)
