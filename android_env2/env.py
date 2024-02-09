import time

import gymnasium as gym

from android_env2.actions import Action, ActionType
from android_env2.config import Settings
from android_env2.phone import Phone
from android_env2.simulator import Simulator
from android_env2.xml_tool import UIXMLTree


class AndroidEnv(gym.Env):
    def __init__(self, config: Settings):
        self.config = config
        self.simulator = Simulator(config)
        self.phone = Phone()
        self.phone.load_from_yaml(config.phone_config_path)
        self.cur_ui_xml_tree = UIXMLTree()
        self.trajectory = None

    def set_traj(self, traj):
        self.trajectory = traj

    def reset(self, **kwargs):
        self.simulator.reset()
        if not self.phone.device_info:
            self.phone.set_device_info(self.simulator.driver.device_info)
        return None, {}

    def step(self, action: Action):
        terminated, truncated = False, False
        if action.action_type != ActionType.FINISH:
            self.simulator.execute_action(action)
        else:
            terminated = True
        return None, None, terminated, truncated, {"action": action}

    def close(self):
        self.simulator.stop_avd()
