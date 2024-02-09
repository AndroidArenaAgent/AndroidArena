import argparse

from agents.tasks import load_tasks_from_files
from android_env2.actions import AndroidActionWrapper
from android_env2.config import get_settings
from android_env2.env import AndroidEnv
from android_env2.observation import MixObsWrapper
from android_env2.reward import DummyRewardWrapper
from agents.replay_agent import ReplayAgent


def get_args():
    args = argparse.ArgumentParser(description='replay_agent')
    args.add_argument('--test_app', default="calendar", type=str, help='test_apps')
    return args.parse_args()


def get_env(reward_lm=None):
    settings = get_settings()
    env = AndroidEnv(settings)

    env = MixObsWrapper(env)
    env = DummyRewardWrapper(env)
    env = AndroidActionWrapper(env)
    return env


def run(args=get_args()):
    replay_agent = ReplayAgent(env=get_env(), args=args)
    task_list = load_tasks_from_files(filename=f"tasks/{args.test_app}.yaml")
    for task in task_list:
        if not task.action_sequence:
            continue
        replay_agent.run(task)


if __name__ == "__main__":
    run()
