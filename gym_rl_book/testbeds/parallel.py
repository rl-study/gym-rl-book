import psutil
import collections
import multiprocessing as mul
import matplotlib.pyplot as plt

TaskSpec = collections.namedtuple('TaskSpec',
                                  [
                                      'env', 'agent', 'parameter',
                                      'episode_count', 'steps_per_episode',
                                      'stats_start_steps'
                                  ])


class PTestbed:
    def __init__(self, num_worker=psutil.cpu_count()):
        self.num_worker = num_worker
        self._results = {}
        self._parameter_sets = []

    @staticmethod
    def _run_agent(task_spec):
        env, agent, p = task_spec.env, task_spec.agent, task_spec.parameter

        key = agent.describe()
        print("Running agent {} with parameter {}".format(key, p))
        avg = 0.0
        count = 0
        for i in range(task_spec.episode_count):
            env.reset()
            agent.reset(p)
            for step in range(task_spec.steps_per_episode):
                act = agent.action()
                _, reward, _, _ = env.step(act)
                agent.learn(act, reward)

                if step >= task_spec.stats_start_steps:
                    count += 1
                    avg = (avg * (count - 1) + reward) / count
            if (i + 1) % 100 == 0:
                print("[Agent {}] Episode: {}".format(key, i + 1))
        return key, avg

    @staticmethod
    def _build_tasks(env_maker, agents_maker, parameter_sets, episode_count, steps_per_episode, stats_start_steps):
        tasks = []
        for maker in agents_maker:
            for p in parameter_sets:
                tasks.append(TaskSpec(env=env_maker(),
                                      agent=maker(),
                                      parameter=p,
                                      episode_count=episode_count,
                                      steps_per_episode=steps_per_episode,
                                      stats_start_steps=stats_start_steps))
        return tasks

    def run(self, env_maker, agents_maker, parameter_sets, episode_count, steps_per_episode, stats_start_steps):
        self._results = {}
        self._parameter_sets = parameter_sets
        tasks = PTestbed._build_tasks(env_maker,
                                      agents_maker,
                                      parameter_sets,
                                      episode_count,
                                      steps_per_episode,
                                      stats_start_steps)

        with mul.Pool(processes=self.num_worker) as pool:
            results = pool.map(PTestbed._run_agent, tasks)

        for key, avg in results:
            self._results[key] = avg

    def plot(self):
        xs = [x for x in range(len(self._parameter_sets))]
        colors = ['r', 'g', 'b', 'c', 'y', 'm', 'k']

        fig, ax1 = plt.subplots(1, 1, figsize=[20, 10])
        ax1.set_xticks(xs)
        ax1.set_xticklabels(self._parameter_sets)
        fig.suptitle("Parameter Study")

        color_idx = 0
        for k in self._results.keys():
            ax1.plot(xs, self._results[k], colors[color_idx], label=k)
            color_idx += 1
            color_idx %= len(colors)

        ax1.legend()

