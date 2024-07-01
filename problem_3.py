import numpy as np
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

class BikeSharingEnv(gym.Env):
    def __init__(self):
        super(BikeSharingEnv, self).__init__()
        self.num_bikes = 1000
        self.num_areas = 10
        self.action_space = gym.spaces.MultiDiscrete([self.num_bikes + 1] * self.num_areas)
        self.observation_space = gym.spaces.Box(low=0, high=self.num_bikes, shape=(self.num_areas,), dtype=np.int32)
        self.state = np.zeros(self.num_areas, dtype=np.int32)
        self.demand = np.array([
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

    def reset(self):
        self.state = np.zeros(self.num_areas, dtype=np.int32)
        return self.state

    def step(self, action):
        self.state = action
        reward = self.calculate_reward()
        done = True
        return self.state, reward, done, {}

    def calculate_reward(self):
        revenue_per_bike = 3 * 365 * 5  # 5年总收入
        cost_per_bike = 4000 + (0.15 * revenue_per_bike)  # 成本+维护费用
        total_revenue = np.sum(self.state * revenue_per_bike)
        total_cost = np.sum(self.state * cost_per_bike)
        return total_revenue - total_cost

# 创建环境
env = DummyVecEnv([lambda: Monitor(BikeSharingEnv())])

# 训练模型
model = PPO("MlpPolicy", env, verbose=1, device='cuda')
model.learn(total_timesteps=10000)

# 测试模型
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    if dones:
        break

print("优化后的单车投放方案:", obs)
