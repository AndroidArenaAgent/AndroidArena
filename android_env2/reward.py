import re
from typing import SupportsFloat, Any

import tiktoken
from colorama import Fore
from gymnasium import Env
from gymnasium.core import ObsType, ActType, Wrapper

from android_env2.actions import ActionType
from android_env2.exception import OutputParserException


class AndroidRewardWrapper(Wrapper[ObsType, ActType, ObsType, ActType]):
    def __init__(self, env: Env[ObsType, ActType]):
        Wrapper.__init__(self, env)

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        if action.action_type == ActionType.FINISH:
            reward = self.reward(action, obs, reward)
        else:
            reward = 0.
        return obs, reward, terminated, truncated, info

    def reward(self, action: ActType, obs: ObsType, reward: SupportsFloat) -> SupportsFloat:
        raise NotImplementedError


class DummyRewardWrapper(AndroidRewardWrapper):

    def reward(self, action, obs, reward):
        # dummy reward, for testing
        return 1.


class RegexMatchRewardWrapper(AndroidRewardWrapper):

    def reward(self, action, obs, reward):
        match_yes = re.search(
            r".*success.*", obs.strip(), re.MULTILINE | re.IGNORECASE | re.DOTALL
        )
        if match_yes:
            return 1.
        match_no = re.search(
            r".*fail.*", obs.strip(), re.MULTILINE | re.IGNORECASE | re.DOTALL
        )
        if match_no:
            return 0.
        return 0.


class ImageMatchRewardWrapper(AndroidRewardWrapper):
    def __init__(self, env, target_img):
        super().__init__(env)
        self.target_img = target_img

    def reward(self, action, obs, reward):
        # check image similarity
        match_score = self.env.simulator.driver.match(self.target_img)["similarity"]
        return match_score


class PromptRewardWrapper(AndroidRewardWrapper):
    def __init__(self, env, reward_lm):
        super().__init__(env)
        self.reward_lm = reward_lm

    def reward(self, action, obs, reward):
        response = self.reward_lm(self.env.trajectory.data)
        print(Fore.MAGENTA + f"LM Reward Function: {response}\n" + Fore.RESET)
        match_yes = re.search(
            r".*Yes.*", response.strip(), re.MULTILINE | re.IGNORECASE | re.DOTALL
        )
        if match_yes:
            return 1.
        match_no = re.search(
            r".*No.*", response.strip(), re.MULTILINE | re.IGNORECASE | re.DOTALL
        )
        if match_no:
            return 0.
        raise OutputParserException("reward parse error.")


class LogRewardWrapper(AndroidRewardWrapper):
    def reward(self, action, obs, reward):
        # todo redirect logcat output to log file
        pass
