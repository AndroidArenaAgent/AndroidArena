import base64
import os.path
import pickle
import re
import time
import traceback
from collections import defaultdict
from io import BytesIO

from agents.tasks import Task

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
        }}

        pre {{
            background-color: #f4f4f4;
            padding: 10px;
        }}

        .expandable {{
            cursor: pointer;
            font-weight: bold;
        }}
        .json-line {{
            border-left: 2px solid #ccc;
            margin-left: 20px;
            padding-left: 10px;
        }}

         .container {{
            width: 100%;
        }}

        .row {{
            display: flex;
            align-items: center;
            margin-bottom: 10px;
        }}


        .image {{
            display: flex;
            justify-content: center;
            align-items: center;
            margin-left: 30px;
            height: 500px;
            width: auto;
        }}

        .green-font {{
            color: green;
        }}
        .yellow-font{{
            color: rgb(255, 183, 0);
        }}
         .red-font{{
            color: rgb(220, 20, 60);
        }}
        .blue-font{{
            color:rgb(0, 47, 255);
        }}
    </style>
</head>
<body>
    <h2>Task</h2>
        <pre>{task}</pre>
        <pre>{constrain}</pre>
    <h2>System Prompt</h2>
        <pre>{system}</pre>
    <h2>Few-shot Examples</h2>
        <pre>{examples}</pre>
    <h2>Previous Reflection</h2>
        <pre>{prev_reflection}</pre>
    <div class="container">
        {body}
    </div>

</body>
</html>

"""


def save_trajectory(folder="trajectory"):
    def decorator(run_func):
        def wrapper(self, *args, **kwargs):
            try:
                run_func(self, *args, **kwargs)
            except Exception as e:
                self.trajectory.exception_str = traceback.format_exc()
                raise e
            finally:
                obs = f"obs_{self.args.hist_steps}_" if self.args.with_obs else ""
                suffix = "_" + self.args.tj_suffix if self.args.tj_suffix else ""
                para_folder = folder + "_" + self.args.model_name + "_" + self.args.agent_type + "_" + obs + self.args.test_app + suffix
                self.trajectory.save_to_html(para_folder)
                self.trajectory.save_to_pkl(para_folder)

        return wrapper

    return decorator


class Trajectory:
    def __init__(self, task: Task):
        self.task = task
        self.data = []
        self.prev_reflection = ""
        self.reflection = None
        self.system_str = ""
        self.example_str = ""
        self.exception_str = ""

    def add(self, state=None, thought=None, action=None, reward=None, reflection=None):
        if thought:
            self.data[-1]["thought"] = thought
        if action:
            self.data[-1]["action"] = action
        if reward is not None:
            self.data[-1]["reward"] = reward
        if reflection:
            # add reflection when episode failed
            self.task.reflection.append(reflection)
            self.reflection = reflection
        if state:
            self.data.append({"state": state})

    def get_last_k(self, k=5):
        return self.data[-k:]

    def save_to_html(self, folder):
        inst = self.task.instruction.replace(" ", "_")
        inst = re.sub(r"[\/,\.@\\\:\*\?\"\<\>\|]", "", inst)
        inst = inst.replace("__", "_")
        inst = inst.replace("__", "_")
        body_str = ""
        for i, data in enumerate(self.data):
            head = f"<h2>Step {i}</h2>"
            obs_str = ""
            if 'state' in data:
                if isinstance(data['state'], str):
                    obs_str += f"<pre>{data['state']}</pre>"
                elif isinstance(data['state'], dict):
                    image = data['state']['image']
                    image_bytes_io = BytesIO()
                    image.save(image_bytes_io, format="JPEG")
                    base64_image = base64.b64encode(image_bytes_io.getvalue()).decode('ascii')
                    obs_str += f"<img class=\"image\" src=\"data:image/png;base64,{base64_image}\">"
                    obs_str += f"<pre>{data['state']['text']}</pre>"
            thought_str = ""
            if 'thought' in data:
                thought_str = f"<br><div class=\"green-font\">Thought: {data['thought']}</div>"
            action_str = ""
            if 'action' in data:
                action_str = f"<br><div class=\"blue-font\">Action: {data['action']}</div>"
            reward_str = ""
            if 'reward' in data:
                reward_str = f"<br><div class=\"red-font\">Reward: {data['reward']}</div>"
            body_str += head + "<div class='row'>" + obs_str + "</div>" + thought_str + action_str + reward_str
        if self.exception_str:
            body_str += f"<h2>Exception</h2><pre>{self.exception_str}</pre>"
        if self.reflection:
            body_str += f"<h2>Reflection</h2><pre>{self.reflection}</pre>"
        if not os.path.exists(folder):
            os.mkdir(folder)
        inst = inst[:100]
        with open(f"{folder}/{inst}_{time.strftime('%m-%d_%H-%M-%S', time.localtime(int(time.time())))}.html",
                  "w", encoding='utf-8') as f:
            f.write(
                HTML_TEMPLATE.format(title=self.task.instruction, task=self.task.instruction,
                                     constrain=self.task.constrain_prompt, system=self.system_str,
                                     examples=self.example_str, prev_reflection=self.prev_reflection, body=body_str))

    def save_to_pkl(self, folder):
        inst = self.task.instruction.replace(" ", "_")
        inst = re.sub(r"[\/,\.@\\\:\*\?\"\<\>\|]", "", inst)
        inst = inst.replace("__", "_")
        inst = inst.replace("__", "_")
        if not os.path.exists(folder):
            os.mkdir(folder)
        inst = inst[:100]
        pickle.dump(
            {"task": self.task.as_dict(), "data": self.data, "reflection": self.reflection,
             "exception": self.exception_str},
            open(f"{folder}/{inst}_{time.strftime('%m-%d_%H-%M-%S', time.localtime(int(time.time())))}.pkl", "wb")
        )


class ReplayBuffer:
    def __init__(self):
        self.exp = defaultdict(list)

    def add_exp(self, instruction, action_sequence, final_state):
        self.exp[instruction].append({"action_sequence": action_sequence,
                                      "final_state": final_state})

    def retrieve_topk(self, instruction, top_k):
        """
        find the top_k most similar experiences
        """
        return

    def save_to_vector_db(self):
        pass

    def save_to_db(self):
        pass

    def save_to_json(self):
        pass
