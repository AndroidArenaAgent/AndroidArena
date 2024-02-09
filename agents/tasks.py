import os
import random
import re
from dataclasses import dataclass, field
from typing import List, Any, Dict, Tuple

import yaml

from agents.prompt import SYSTEM_TEMPLATE, EXAMPLE_PROMPT, ACT_TEMPLATE, REWARD_PROMPT, EXAMPLES, REFLECTION_PROMPT


@dataclass
class Task:
    instruction: str = "do not need to do anything."
    obs_type: str = "text"
    reward_type: str = "dummy"
    hist_steps: int = 3

    examples: List[Tuple[str, str]] = field(default_factory=list)  # few-shot examples
    reflection_examples: List[Tuple[str, str]] = field(default_factory=list)  # self reflection examples
    action_sequence: List[Dict] = field(default_factory=list)
    reflection: List[str] = field(default_factory=list)  # reflection

    target_img: str | None = None
    regex: str | None = None
    success: bool = False

    system_prompt: str = SYSTEM_TEMPLATE
    example_prompt: str = EXAMPLE_PROMPT
    act_prompt: str = ACT_TEMPLATE
    constrain_prompt: str = ""
    reflection_prompt: str = REFLECTION_PROMPT
    reward_prompt: str = REWARD_PROMPT

    max_step: int = 30
    max_repeat_step: int = 5
    exe_if_failed: bool = False
    meta_data: dict[str, Any] = field(default_factory=dict)

    def as_dict(self):
        return {"task": self.instruction, "action_sequence": self.action_sequence, "reflection": self.reflection,
                "target_img": self.target_img, "regex": self.regex, "exe_if_failed": self.exe_if_failed}

    @classmethod
    def load_from_yaml(cls, path):
        data = yaml.safe_load(open(path, "r"))
        task = cls()
        if "instruction" in data:
            task.instruction = data["instruction"]
        if "reward_type" in data:
            task.reward_type = data["reward_type"]
        task.examples = EXAMPLES
        return task

    def save_to_yaml(self):
        pass


def ui2code_to_dict(code_list: List[str]):
    action_dict_list = []
    for c in code_list:
        act_dict = {}
        if "app_start" in c:
            package = re.findall(r"'(.+?)'", c)[0]
            act_dict = {"action": "START_APP", "package": package}
        if "xpath" in c:
            xpath = re.findall(r"xpath\('(.+?)'\)", c)[0]
            act_dict["xpath"] = xpath
            if "long_click()" in c:
                act_dict["action"] = "LONG_CLICK"
            elif "click()" in c:
                act_dict["action"] = "CLICK"
            elif "set_text(" in c:
                act_dict["action"] = "SET_TEXT"
                text = re.findall(r"set_text\('(.+?)'\)", c)[0]
                act_dict["text"] = text
        if "swipe_ext" in c:
            direction = re.findall(r"swipe_ext\('(.+?)'\)", c)[0]
            act_dict = {"action": f"swipe_{direction}".upper()}
        if "press" in c:
            act = re.findall(r"press\('(.+?)'\)", c)[0]
            act_dict = {"action": f"press_{act}".upper()}
        action_dict_list.append(act_dict)
    action_dict_list.append({"action": "FINISH", "text": ""})
    return action_dict_list


def load_tasks_from_files(folder=None, filename=None) -> List[Task]:
    task_list = []
    file_list = []
    if filename:
        file_list = [filename]
    else:
        for root, ds, fs in os.walk(folder):
            for f in fs:
                fullname = os.path.join(root, f)
                file_list.append(fullname)
    for fn in file_list:
        data = yaml.safe_load(open(fn, "r"))
        random_suffix = ""
        if "slack" in fn:
            random_suffix = random.randint(0, 100)
        for ins in data["tasks"]:
            task = Task()
            task.instruction = ins["instruction"]
            task.instruction = task.instruction.replace("myspace", "myspace" + str(random_suffix))
            task.instruction = task.instruction.replace("myproject", "myproject" + str(random_suffix))
            task.instruction = task.instruction.replace("work_channel", "work_channel" + str(random_suffix))
            task.obs_type = ins["obs_type"] if "obs_type" in ins else data["obs_type"]
            task.reward_type = ins["reward_type"] if "reward_type" in ins else data["reward_type"]
            task.max_step = ins["max_step"] if "max_step" in ins else data["max_step"]
            task.max_repeat_step = ins["max_repeat_step"] if "max_repeat_step" in ins else data["max_repeat_step"]
            task.exe_if_failed = ins["exe_if_failed"] if "exe_if_failed" in ins else False
            task.constrain_prompt = ins["constrains"] if "constrains" in ins else ""
            if "action_seq" in ins:
                task.action_sequence = ui2code_to_dict(ins["action_seq"])
            task.examples = EXAMPLES
            task_list.append(task)
    return task_list
