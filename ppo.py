import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from bicycle_iter import *
from mat2d_round import *


total_days = 365 * 5

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
        self.epoch = 0
        self.best_dist = np.zeros(10, dtype=int)
        self.best_profit = 0

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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.bike_allocation = np.zeros((10,)) #np.full(10, self.initial_bikes // 10)
        self.total_profit = 0
        observation = self._get_observation()
        info = {}  # No specific information to return
        return observation, info

    def step(self, action):
        self.epoch += 1
        # Ensure action is within the valid range and is an integer
        action = np.clip(action, 0, None)*1000
        action = mat2d_round(action.reshape((-1, 1)), 0).flatten()
        old_action = action.copy()
        # action += np.array([126, 134, 118, 143, 124, 127, 120, 113, 173, 133])
        action += np.array([ 198, 149, 297, 143, 323, 127, 120, 1188, 428, 583])
        # action += np.ones_like(action, dtype=int)
        self.bike_allocation = action
        # print(action)

        # Calculate rewards
        # revenue = 0
        # cost = 0
        # for i in range(10):
        #     for j in range(10):
        #         trips = min(self.bike_allocation[i], self.demand_matrix[i, j])
        #         sunny_days_time = (self.sunny_days / (self.sunny_days + self.rainy_days)) * total_days
        #         revenue += trips * self.order_income * sunny_days_time * self.sunny_hours
        #         cost += trips * self.maintenance_cost * self.order_income

        # fixed_annual_cost = self.fixed_cost + self.initial_bikes * self.bike_cost / self.bike_life * (
                    # 1 - self.scrap_rate)
        # total_cost = cost + fixed_annual_cost
        # print("total_cost", total_cost)
        # print("revenue", revenue)

        demand_matrix = (self.demand_matrix / 6).round(0).astype(int)
        profit = total_revenue(action, demand_matrix, mute = True)
        self.total_profit = profit

        # Determine if the episode is done
        done = False  # No specific termination condition for now
        truncated = False  # No specific truncation condition for now

        reward = profit  # Reward is the profit
        print(f"epoch{self.epoch:4d} profit: {profit:.0f}\tdelta: {old_action}, init: {action}")
        if profit > self.best_profit:
            self.best_profit = profit
            self.best_dist = action
        return self._get_observation(), reward, done, truncated, {}

    def _get_observation(self):
        return self.bike_allocation.astype(np.float32)

    def render(self, mode='human'):
        print(f"Bike allocation: {self.bike_allocation}, Total profit: {self.total_profit}")


# Instantiate the environment
env = BikeSharingEnv()

# Create the model
model = PPO("MlpPolicy", env, verbose=1, learning_rate=1e-7, use_sde=True)

# Train the model
model.learn(total_timesteps=100000)

# Save the model
# model.save("ppo_bike_sharing")

# Test the trained model
obs, _ = env.reset()  # Unpack observation and info
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    if done or truncated:
        break

print(env.best_dist)
print(env.best_profit)