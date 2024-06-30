import gym
from gym import spaces
import numpy as np


class BikeSharingEnv(gym.Env):
    def __init__(self, initial_bikes=1000, maintenance_cost=0.165, order_income=3, bike_cost=4000, fixed_cost=500000,
                 bike_life=5, scrap_rate=0.1, sunny_hours=12, rainy_hours=0, sunny_days=10.19, rainy_days=5.2):
        super(BikeSharingEnv, self).__init__()

        # Constants
        self.initial_bikes = initial_bikes
        self.maintenance_cost = maintenance_cost
        self.order_income = order_income
        self.bike_cost = bike_cost
        self.fixed_cost = fixed_cost
        self.bike_life = bike_life
        self.scrap_rate = scrap_rate
        self.sunny_hours = sunny_hours
        self.rainy_hours = rainy_hours
        self.sunny_days = sunny_days
        self.rainy_days = rainy_days

        # Action space: number of bikes to allocate to each region
        self.action_space = spaces.Box(low=0, high=initial_bikes, shape=(10,), dtype=np.int32)

        # Observation space: demand in each region
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(10,), dtype=np.float32)

        # Load the demand data from the document
        self.demand_matrix = np.array([
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

        self.reset()

    def reset(self):
        self.bike_allocation = np.full(10, self.initial_bikes // 10)
        return self._get_observation()

    def step(self, action):
        # Update bike allocation based on action
        self.bike_allocation = action

        # Calculate rewards
        revenue = 0
        cost = 0
        for i in range(10):
            for j in range(10):
                trips = min(self.bike_allocation[i], self.demand_matrix[i, j])
                revenue += trips * self.order_income
                cost += trips * self.maintenance_cost * self.order_income

        total_cost = cost + self.fixed_cost + self.initial_bikes * self.bike_cost / self.bike_life * (
                    1 - self.scrap_rate)
        profit = revenue - total_cost

        # Determine if the episode is done
        done = False  # For simplicity, we can define the termination condition as you see fit

        reward = profit  # Reward is the profit

        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        return self.bike_allocation

    def render(self, mode='human'):
        print(f"Bike allocation: {self.bike_allocation}")


# Instantiate the environment
env = BikeSharingEnv()


