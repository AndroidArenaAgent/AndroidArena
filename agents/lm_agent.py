import traceback

import time
from typing import List, Tuple, Dict, Any

from colorama import Fore
from gymnasium import Env
from gymnasium.core import ObsType, ActType
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate, SystemMessagePromptTemplate, \
    HumanMessagePromptTemplate
from langchain.schema import SystemMessage
from uiautomator2 import UiObjectNotFoundError
from uiautomator2.exceptions import XPathElementNotFoundError

from agents.action_parser import AgentOutputParser
from agents.agent_base import BaseAgent
from agents.prompt import (REFLECTION_HEADER,
                           EXAMPLE_PROMPT,
                           SYSTEM_PROMPT,
                           ACT_PROMPT,
                           REFLECTION_PROMPT_SYSTEM,
                           REFLECTION_PROMPT, CONSTRAIN_SYSTEM_HEADER)
from agents.replay_agent import ReplayAgent
from agents.replay_buffer import Trajectory, save_trajectory
from agents.tasks import Task
from agents.utils import load_llm_agent, truncate_scratchpad, load_tokenizer
from android_env2.actions import Action, ActionType
from android_env2.exception import AndroidActionException, OutputParserException


class LMAgent(BaseAgent):
    def __init__(self, env: Env[ObsType, ActType], args):
        super().__init__(env, args)
        assert args.agent_type in ["direct", "react", "react_reflection"]
        self.chat_model = load_llm_agent(args.model_provider, args.temperature)
        self.tokenizer = load_tokenizer(model_name=args.model_name)
        self.action_parser = AgentOutputParser()
        self.action_repeat_cnt = 0
        self.cur_step = 1
        self.replay_agent = ReplayAgent(env, args)

    def create_agent_prompt(self, stage: str):
        task = self.trajectory.task
        app_string = "\n".join(
            [f"> {app.name}: {app.description}" for package, app in self.env.phone.apps.items()]
        )
        date = time.strftime('%b %d %Y %A', time.localtime(int(time.time())))

        reflection = ""
        if "react_reflection" == self.args.agent_type and task.reflection:
            reflection = REFLECTION_HEADER + 'Reflections:\n- ' + '\n- '.join([r for r in task.reflection])
            if not self.trajectory.prev_reflection:
                self.trajectory.prev_reflection = reflection

        constrain = ""
        if "constrain" == self.args.test_app:
            constrain = CONSTRAIN_SYSTEM_HEADER + f'\nConstrain: {task.constrain_prompt}'

        instruction = task.instruction
        scratchpad, still_exceed = self._construct_react_scratchpad(self.trajectory.get_last_k(self.args.hist_steps),
                                                                    stage)

        if still_exceed:  # remove few-shot examples
            chat_prompt_template = ChatPromptTemplate.from_messages(
                [SystemMessagePromptTemplate(prompt=SYSTEM_PROMPT),
                 HumanMessagePromptTemplate(prompt=ACT_PROMPT)]
            )
        else:
            chat_prompt_template = ChatPromptTemplate.from_messages(
                [SystemMessagePromptTemplate(prompt=SYSTEM_PROMPT),
                 FewShotChatMessagePromptTemplate(example_prompt=EXAMPLE_PROMPT, examples=task.examples),
                 HumanMessagePromptTemplate(prompt=ACT_PROMPT)]
            )

        message = chat_prompt_template.format_prompt(app_string=app_string, date=date, reflection=reflection,
                                                     constrain=constrain, instruction=instruction,
                                                     scratchpad=scratchpad).to_messages()

        if not self.trajectory.system_str:
            self.trajectory.system_str = message[0].content

        if not self.trajectory.example_str:
            self.trajectory.example_str = "\n".join([m.content for m in message[1:-1]])

        print(Fore.CYAN + f"Prompt: {message[-1].content}" + Fore.RESET, end="\n\n")
        return message

    def create_reflection_prompt(self):
        task = self.trajectory.task
        scratchpad, still_exceed = self._construct_react_scratchpad(
            self.trajectory.get_last_k(len(self.trajectory.data)), stage="Reflection")
        if still_exceed:
            scratchpad = self.tokenizer.decode(self.tokenizer.encode(scratchpad)[-self.args.scratchpad_length:])
        reflection_prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=REFLECTION_PROMPT_SYSTEM),
            HumanMessagePromptTemplate(prompt=REFLECTION_PROMPT)]
        )

        previous_reflection = ""
        if task.reflection:
            previous_reflection = 'Previous Reflections:\n- ' + '\n- '.join([r for r in task.reflection])

        message = reflection_prompt_template.format_prompt(instruction=task.instruction,
                                                           scratchpad=scratchpad,
                                                           previous_reflection=previous_reflection).to_messages()
        return message

    def _construct_react_scratchpad(
            self, intermediate_steps: List[Dict[str, Any]], stage: str
    ) -> (str, bool):
        """Construct the scratchpad that lets the agent continue its think, action and reflection process."""
        scratchpad = ""
        for t, step in enumerate(intermediate_steps[:-1]):
            scratchpad += f"Step {self.cur_step - len(intermediate_steps) + 1 + t}:\n"
            if self.args.with_obs:
                scratchpad += f"\nPrevious Observation: {step['state']['text'] if isinstance(step['state'], dict) else step['state']}\n"
            scratchpad += f"\nPrevious Action: {step['action']}\n\n"
        scratchpad += f"Step {self.cur_step}:\n"
        last_step = intermediate_steps[-1]
        state = last_step['state']['text'] if isinstance(last_step['state'], dict) else last_step['state']
        if stage in ["Think", "Reflection"]:
            scratchpad += f"\nObservation: '{state}\n{stage}: "
        elif stage == "Action":
            assert len(last_step) == 2
            scratchpad += f"\nObservation: {state}\nThought: {last_step['thought']}\n{stage}: "

        scratchpad, still_exceed = truncate_scratchpad(scratchpad, n_tokens=self.args.scratchpad_length)
        return scratchpad, still_exceed

    def _construct_direct_scratchpad(
            self, intermediate_steps: List[Tuple[Action, str]], stage: str
    ) -> str:
        """Construct the scratchpad that lets the agent continue its think, action and reflection process."""
        scratchpad = ""
        for state, thought, action in intermediate_steps[:-1]:
            scratchpad += f"\nPrevious Observation: {state}\nPrevious Action: {action}\n\n"
        last_step = intermediate_steps[-1]
        scratchpad_suffix = f"\nPrevious Observation: {last_step[0]}\n{stage}: "
        scratchpad += scratchpad_suffix
        max_hist_length = getattr(self.args, "max_hist_length", None)
        if max_hist_length:
            scratchpad = self.tokenizer.decode(self.tokenizer.encode(scratchpad)[-max_hist_length:])
        return scratchpad

    def reflection(self):
        reflection = self.chat_model(self.create_reflection_prompt()).content
        print(Fore.LIGHTGREEN_EX + f"Reflection: {reflection}\n\n" + Fore.RESET)
        self.trajectory.add(reflection=reflection)

    def check_repeat_action(self, action):
        last_action = self.trajectory.data[-2]["action"] if len(self.trajectory.data) > 1 else None
        if last_action and action == last_action:
            self.action_repeat_cnt += 1
            if self.action_repeat_cnt > self.trajectory.task.max_repeat_step:
                raise ValueError(f"Exceed max {self.action_repeat_cnt} repeat action {action}")
            return True
        else:
            self.action_repeat_cnt = 0
            return False

    def select_action(self):
        # think
        think_response = self.chat_model(self.create_agent_prompt(stage="Think")).content
        print(Fore.GREEN + f"Think: {think_response}" + Fore.RESET, end="\n\n")
        self.trajectory.add(thought=think_response)

        action_response = self.action_parser.parse(think_response)
        return action_response

    @save_trajectory(folder=f"traj")
    def run(self, task: Task):
        self._reset_agent()
        print(Fore.RED + f"Task: {task.instruction}" + Fore.RESET, end="\n\n")
        self.trajectory = Trajectory(task=task)
        self.env.set_traj(self.trajectory)
        obs, info = self.env.reset()
        print(Fore.YELLOW + f"Obs: {obs['text'] if isinstance(obs, dict) else obs}" + Fore.RESET, end="\n\n")
        self.trajectory.add(state=obs)
        try:
            while not self.terminated:
                try:
                    action = self.select_action()
                    print(Fore.BLUE + f"Action: {action}" + Fore.RESET, end="\n\n")
                    obs, reward, self.terminated, truncated, info = self.env.step(action)
                    self.trajectory.add(action=info["action"])
                except (AndroidActionException,
                        UiObjectNotFoundError, XPathElementNotFoundError,
                        OutputParserException) as e:
                    if isinstance(e, UiObjectNotFoundError) or isinstance(e, XPathElementNotFoundError):
                        e = f"Invalid node id."
                    if isinstance(obs, dict):
                        obs = {"text": str(e), "image": obs["image"]}
                    else:
                        obs = str(e)
                    reward = 0.
                    invalid_action = Action()
                    invalid_action.action_type = ActionType.INVALID
                    self.trajectory.add(action=invalid_action)
                print(Fore.YELLOW + f"Obs: {obs['text'] if isinstance(obs, dict) else obs}" + Fore.RESET, end="\n\n")
                self.trajectory.add(state=obs, reward=reward)
                self.cur_step += 1
                if self.cur_step >= task.max_step:
                    raise ValueError(f"Exceed max step ({task.max_step}) limit, exit.")
        except Exception:
            print("Other exception: ", traceback.format_exc())
        finally:
            if self.terminated and (
                    self.trajectory.data[-2]["reward"] == 1. or
                    "reward" in self.trajectory.data[-1] and self.trajectory.data[-1]["reward"] == 1.):
                task.success = True
            else:
                if "react_reflection" == self.args.agent_type:
                    self.reflection()
                elif task.exe_if_failed:
                    print(Fore.RED + "LM Agent failed, executing Replay Agent" + Fore.RESET)
                    self.replay_agent.run(task)
