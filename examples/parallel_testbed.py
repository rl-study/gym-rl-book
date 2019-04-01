import gym
import numpy as np
import gym_rl_book
from gym_rl_book.testbeds.parallel import PTestbed


class EpsilonGreedyAgent:
    def __init__(self, epsilon=0, step_size=None):
        self.epsilon = epsilon
        self.step_size = step_size

        self.reset()

    def describe(self):
        return "EpsilonGreedyAgent[Epsilon={}][StepSize={}]".format(self.epsilon,
                                                                    self.step_size if self.step_size else "NonConstant")

    def reset(self, epsilon=None, K=10):
        self.q = [0] * K
        self.n = [0] * K
        self.K = K
        if epsilon is not None:
            if epsilon > 1:
                epsilon = 1.
            self.epsilon = epsilon

    def action(self):
        q = np.array(self.q)
        return np.random.choice(np.flatnonzero(q == q.max()) if np.random.uniform() >= self.epsilon else self.K)

    def learn(self, action, reward):
        self.n[action] += 1
        step_size = self.step_size if self.step_size else (1 / self.n[action])
        self.q[action] += (reward - self.q[action]) * step_size


class OptInitAgent:
    def __init__(self, alpha=0.1, q_0=5, K=10):
        self._alpha = 0.1
        self.reset(q_0, K)

    def describe(self):
        return "OptInitAgent[Alpha={}]".format(self._alpha)

    def reset(self, q_0=None, K=10):
        if q_0 is not None:
            self._q_0 = q_0
        self._q = [self._q_0] * K

    def action(self):
        return np.argmax(self._q)

    def learn(self, action, reward):
        self._q[action] += self._alpha * (reward - self._q[action])


class UCBAgent:
    def __init__(self, c=2, K=10):
        self.reset(c, K)

    def describe(self):
        return "UCBAgent"

    def reset(self, c=None, K=10):
        if c is not None:
            self._c = c
        self._q = [0] * K
        self._count = [0] * K
        self._t = 0

    def action(self):
        vals = [q + (self._c * np.sqrt(np.log(self._t) / self._count[idx]) if self._count[idx] > 0 else 10000) for
                idx, q in enumerate(self._q)]
        return np.argmax(vals)

    def learn(self, action, reward):
        self._t += 1
        self._count[action] += 1
        self._q[action] += (reward - self._q[action]) / self._t


class GradientAgent:
    def __init__(self, alpha=1.0, K=10):
        self.reset(alpha, K)

    def describe(self):
        return "GradientAgent"

    def reset(self, alpha=None, K=10):
        if alpha is not None:
            self._alpha = alpha
        self._K = K
        self._pref = [0] * K
        self._pr = np.exp(self._pref) / np.sum(np.exp(self._pref))
        self._reward_avg = 0
        self._t = 0

    def action(self):
        return np.random.choice(self._K, p=self._pr)

    def learn(self, action, reward):
        self._t += 1
        self._reward_avg += (reward - self._reward_avg) / self._t
        for a in range(self._K):
            if a == action:
                self._pref[a] += self._alpha * (reward - self._reward_avg) * (1 - self._pr[action])
            else:
                self._pref[a] -= self._alpha * (reward - self._reward_avg) * self._pr[action]
        self._pr = np.exp(self._pref) / np.sum(np.exp(self._pref))


def main():
    testbed = PTestbed(num_worker=8) # num_worker defaults to logical cpu core count
    env_maker = lambda: gym.make('MultiArmBandit-v0')
    agents_maker = [lambda: EpsilonGreedyAgent(), lambda: OptInitAgent(), lambda: UCBAgent(), lambda: GradientAgent()]
    episode_count, run_steps, stats_start_steps, parameter_sets = 2000, 1000, 0, [1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4]

    testbed.run(env_maker=env_maker,
                agents_maker=agents_maker,
                parameter_sets=parameter_sets,
                episode_count=episode_count,
                steps_per_episode=run_steps,
                stats_start_steps=stats_start_steps)
    testbed.plot()


if __name__ == '__main__':
    main()
