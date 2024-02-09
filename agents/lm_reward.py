import tiktoken
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage

from agents.prompt import REWARD_SYSTEM, REWARD_PROMPT
from agents.tasks import Task
from agents.utils import load_llm_agent, load_tokenizer


class RewardLLM:
    def __init__(self, args):
        self.args = args
        self.chat_model = load_llm_agent(args.model_provider, args.temperature)
        self.instruction = ""
        self.prompt_template = ""
        self.tokenizer = load_tokenizer(args.model_name)

    def set_task(self, task: Task):
        self.instruction = task.instruction
        self.prompt_template = task.reward_prompt

    def construct_prompt(self, traj):
        prompt = ""
        i = len(traj)
        for d in traj[::-1]:
            state = d['state']["text"] if isinstance(d["state"], dict) else d['state']
            if "action" in d:
                cur_prompt = f"Step {i - 1}:\n\nPrevious Observation: {state}\nAction: {d['action']}\n\n"
            else:
                cur_prompt = f"Step {i - 1}:\n\nPrevious Observation: {state}\n\n"
            if len(self.tokenizer.encode(cur_prompt + prompt)) > 3500:
                return prompt
            prompt = cur_prompt + prompt
            i -= 1
        return prompt

    def __call__(self, traj, goal=None):
        chat_prompt = ChatPromptTemplate.from_messages(
            [SystemMessage(content=REWARD_SYSTEM), HumanMessagePromptTemplate(prompt=REWARD_PROMPT)])
        message = chat_prompt.format_prompt(goal=self.instruction if not goal else goal,
                                            traj=self.construct_prompt(traj)).to_messages()
        return self.chat_model(message).content
