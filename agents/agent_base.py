from gymnasium import Env
from gymnasium.core import ObsType, ActType

from agents.replay_buffer import Trajectory
from agents.tasks import Task


class BaseAgent:
    def __init__(self, env: Env[ObsType, ActType], args):
        self.env = env
        self.args = args
        self.trajectory = None
        self.terminated = False
        self.cur_step = 1

    def _reset_agent(self):
        self.cur_step = 1
        self.trajectory = None
        self.terminated = False

    def select_action(self) -> ActType:
        pass

    def learn(self):
        pass

    def run(self, task: Task):
        self.trajectory = Trajectory(task=task)
        obs, info = self.env.reset()
        self.trajectory.add(state=obs)
        while not self.terminated:
            action = self.select_action()
            self.trajectory.add(action=action)
            next_obs, reward, self.terminated, truncated, info = self.env.step(action)
            self.trajectory.add(state=next_obs, reward=reward)
