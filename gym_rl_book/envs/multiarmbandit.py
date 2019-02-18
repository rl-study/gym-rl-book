import gym
from gym import spaces
import numpy as np


class MultiArmBanditEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.K = 10
        self.mean = 0.0
        self.var = 1.0
        self.reward_var = 1.0
        self.bandit_mean = []
        self.stationary = True

        self.action_space = spaces.Discrete(self.K)

        self.reset()

    def tune(self, stationary=True):
        self.stationary = stationary

        self.reset()

    def step(self, action):
        if not self.stationary:
            walk = np.random.normal(0, 0.01, self.K)
            self.bandit_mean += walk

        return None, np.random.normal(self.bandit_mean[action], self.reward_var, 1)[0], False, None

    def reset(self):
        if self.stationary:
            self.bandit_mean = np.random.normal(self.mean, self.var, self.K)
        else:
            self.bandit_mean = np.zeros(self.K)

    def render(self, mode='human'):
        print('Mean: {}, Variance: {}, Reward Variance: {}'.format(self.bandit_mean, self.var, self.reward_var))

    def optimal_choice(self):
        return np.argmax(self.bandit_mean)
