import gym
from gym import spaces
import numpy as np


class MultiArmBanditEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.K = 10
        self.bandit_mean = []
        self.stationary = True

        self.action_space = spaces.Discrete(self.K)

        self.reset()

    def tune(self, K = 10, stationary=True):
        self.stationary = stationary
        self.K = K

        self.reset()

    def reset(self):
        if self.stationary:
            self.bandit_mean = np.random.randn(self.K)
        else:
            self.bandit_mean = np.zeros(self.K)

    def step(self, action):
        if not self.stationary:
            walk = np.random.normal(0, 0.01, self.K)
            self.bandit_mean += walk

        return None, np.random.randn() + self.bandit_mean[action], False, None

    def render(self, mode='human'):
        print('Mean: {}'.format(self.bandit_mean))

    def optimal_choice(self):
        return np.argmax(self.bandit_mean)
