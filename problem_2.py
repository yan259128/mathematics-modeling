import numpy as np
from simanneal import Annealer

# 初始化参数
cost_per_bike = 4000
fixed_annual_cost = 500000
revenue_per_order = 3
maintenance_cost_percentage = 0.165  # 平均值
bike_lifetime = 5  # 年
scrap_value_percentage = 0.10
initial_bikes = 1000

# 区域需求矩阵 (从文档中提取的示例数据)
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


class BikeAllocationProblem(Annealer):

    def __init__(self, state):
        super(BikeAllocationProblem, self).__init__(state)

    def move(self):
        i = np.random.randint(0, 10)
        j = np.random.randint(0, 10)

        if i != j and self.state[i] > 0:
            change = np.random.randint(1, min(self.state[i], 100) + 1)
            self.state[i] -= change
            self.state[j] += change

    def energy(self):
        total_bikes = sum(self.state)
        total_revenue = 0

        for i in range(10):
            for j in range(10):
                total_revenue += demand_matrix[i, j] * revenue_per_order * self.state[i] * 3

        annual_revenue = total_revenue * (10.19 / 15.39)  # 计算年收入，考虑晴天和雨天比例
        total_annual_cost = total_bikes * cost_per_bike + fixed_annual_cost + total_revenue * maintenance_cost_percentage
        scrap_value = total_bikes * cost_per_bike * scrap_value_percentage

        five_year_profit = 5 * (annual_revenue - total_annual_cost) + scrap_value
        return -five_year_profit  # 因为Annealer是最小化能量，我们需要最大化利润，所以取负值


# 初始化参数
initial_allocation = [100] * 10  # 初始投放方案，每个区域100台

# 创建并运行模拟退火算法
bike_problem = BikeAllocationProblem(initial_allocation)
bike_problem.Tmax = 25000  # 初始温度
bike_problem.Tmin = 0.1  # 最小温度
bike_problem.steps = 50000  # 总步数
bike_problem.updates = 100  # 更新步数

optimal_allocation, optimal_profit = bike_problem.anneal()

# 打印结果
print("最佳投放方案:", optimal_allocation)
print("五年后的最大盈利:", -optimal_profit)
