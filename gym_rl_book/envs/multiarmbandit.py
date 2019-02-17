import gym
from gym import spaces
import numpy as np


class MultiArmBanditEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, K=10, mean=0, var=1, reward_var=1):
        self.K = K
        self.mean = mean
        self.var = var
        self.reward_var = reward_var
        self.bandit_mean = []

        self.action_space = spaces.Discrete(K)

        self.reset()

    def step(self, action):
        return None, np.random.normal(self.bandit_mean[action], self.reward_var, 1)[0], False, None

    def reset(self):
        self.bandit_mean = np.random.normal(self.mean, self.var, self.K)

    def render(self, mode='human'):
        print('Mean: {}, Variance: {}, Reward Variance: {}'.format(self.bandit_mean, self.var, self.reward_var))

    def best(self):
        return np.argmax(self.bandit_mean)
