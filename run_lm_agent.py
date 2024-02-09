import argparse
import os

from colorama import Fore
from dotenv import load_dotenv

from agents.lm_agent import LMAgent
from agents.lm_reward import RewardLLM
from agents.replay_agent import ReplayAgent
from agents.tasks import load_tasks_from_files
from android_env2.actions import AndroidActionWrapper
from android_env2.config import get_settings
from android_env2.env import AndroidEnv
from android_env2.observation import MixObsWrapper
from android_env2.reward import PromptRewardWrapper

load_dotenv(".env")


def get_args():
    args = argparse.ArgumentParser(description='lm_agent')
    args.add_argument('--model_provider', default="azure_openai", type=str, help='{openai, azure_openai, llama}')
    args.add_argument('--model_name', default="gpt-35-turbo", type=str, help='{gpt-35-turbo, gpt-4, llama70b}')
    args.add_argument('--agent_type', default="react", type=str, help='{direct, react, react_reflection}')
    args.add_argument('--max_reflection', default=1, type=int, help='max reflection time')
    args.add_argument('--hist_steps', default=5, type=int, help='hist_steps')
    args.add_argument('--mode', default="chat", type=str, help='{chat, completion}')
    args.add_argument('--temperature', default=0.1, type=float, help='temperature')
    args.add_argument('--max_tokens', default=2000, type=int, help='max_tokens')
    args.add_argument('--stop_token', default=None, type=list, help='stop_token')
    args.add_argument('--with_obs', action="store_true", help='with_obs')
    args.add_argument('--scratchpad_length', default=2000, type=int, help='scratchpad_length')
    args.add_argument('--test_app', default="calendar", type=str, help='test_apps')
    args.add_argument('--tj_suffix', default="", type=str, help='tj_suffix')
    return args.parse_args()


def get_env(reward_lm=None):
    settings = get_settings()
    env = AndroidEnv(settings)

    env = MixObsWrapper(env)
    env = PromptRewardWrapper(env, reward_lm)
    env = AndroidActionWrapper(env)
    return env


def run():
    args = get_args()
    if args.model_provider == "azure_openai":
        os.environ["AZURE_ENGINE"] = args.model_name
    if args.model_provider == "llama":
        llama_engine_dict = {"llama70b": "llama-2-70b-chat", "llama13b": "llama-2-13b-chat"}
        os.environ["LLAMA_ENGINE"] = llama_engine_dict[args.model_name]
    reward_lm = RewardLLM(args)
    lm_agent = LMAgent(env=get_env(reward_lm), args=args)
    replay_agent = ReplayAgent(env=get_env(), args=args)
    task_list = load_tasks_from_files(filename=f"tasks/{args.test_app}.yaml")
    for task in task_list:
        reward_lm.set_task(task)
        lm_agent.run(task)
        success = task.success
        reflection_cnt = 1
        while not success and "react_reflection" == args.agent_type and reflection_cnt <= args.max_reflection:
            lm_agent.run(task)
            success = task.success
            reflection_cnt += 1
        if "react_reflection" == args.agent_type and task.exe_if_failed and not success:
            print(Fore.RED + "LM Agent failed, executing Replay Agent" + Fore.RESET)
            replay_agent.run(task)


if __name__ == "__main__":
    run()
