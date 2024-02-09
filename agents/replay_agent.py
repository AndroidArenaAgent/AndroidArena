from colorama import Fore

from agents.agent_base import BaseAgent
from agents.replay_buffer import Trajectory, save_trajectory


class ReplayAgent(BaseAgent):
    def __init__(self, env, args):
        super().__init__(env, args)
        pass

    @save_trajectory(folder="tj_replay")
    def run(self, task):
        self._reset_agent()
        print(Fore.RED + f"Task: {task.instruction}" + Fore.RESET, end="\n\n")
        self.trajectory = Trajectory(task=task)
        obs, info = self.env.reset()
        print(Fore.YELLOW + f"Obs: {obs['text'] if isinstance(obs, dict) else obs}" + Fore.RESET, end="\n\n")
        self.trajectory.add(state=obs)
        for action in task.action_sequence[:-1]:
            print(Fore.BLUE + f"Action: {action}" + Fore.RESET, end="\n\n")
            self.trajectory.add(action=action)
            obs, reward, terminated, truncated, info = self.env.step(action)
            print(Fore.YELLOW + f"Obs: {obs['text'] if isinstance(obs, dict) else obs}" + Fore.RESET, end="\n\n")
            self.trajectory.add(state=obs, reward=reward)
            self.cur_step += 1
