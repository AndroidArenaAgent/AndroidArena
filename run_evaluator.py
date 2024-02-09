import json
import traceback
from collections import defaultdict

import pandas as pd
import spacy
import yaml
from dotenv import load_dotenv

import difflib
import os
import pickle
import re

from langchain.chat_models import AzureChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

from agents.prompt import REWARD_SYSTEM, REWARD_PROMPT
from agents.tasks import Task

from agents.action_parser import AgentOutputParser
from agents.utils import load_tokenizer

load_dotenv(".env")
nlp = spacy.load("en_core_web_md")


def is_same_action(a1, a2) -> bool:
    if a1["action"] != a2["action"]:
        return False
    if "START_APP" == a1["action"]:
        return True if a1["package"] == a2["package"] else False
    if "CLICK" in a1["action"]:
        match_nearby = ("nearby_xpath" in a1 and a2["xpath"] in a1["nearby_xpath"]) or (
                "nearby_xpath" in a2 and a1["xpath"] in a2["nearby_xpath"])
        return True if a1["xpath"] == a2["xpath"] or match_nearby else False
    if "SET_TEXT" == a1["action"]:
        v1 = nlp(a1["text"])
        v2 = nlp(a2["text"])
        try:
            text_match = v1.similarity(v2) >= 0.6
        except UserWarning:
            text_match = difflib.SequenceMatcher(None, a1["text"], a2["text"]).quick_ratio() >= 0.6
        match_nearby = ("nearby_xpath" in a1 and a2["xpath"] in a1["nearby_xpath"]) or (
                "nearby_xpath" in a2 and a1["xpath"] in a2["nearby_xpath"])
        return True if (a1["xpath"] == a2["xpath"] or match_nearby) and text_match else False
    return True


def prepare_eval_data(traj_folder, filename=None, reflection_cnt=0, all_trace=False, self_agent_rw=False, step=None):
    file_list = []
    if filename:
        file_list = [filename]
    else:
        for root, ds, fs in os.walk(traj_folder):
            for f in fs:
                if f.endswith(".pkl"):
                    fullname = os.path.join(root, f)
                    file_list.append(fullname)

    if len(file_list) == 0:
        raise FileNotFoundError(f"Empty folder {traj_folder}.")

    def lcs(s1, s2):
        m = [[0 for _ in range(len(s2) + 1)] for _ in range(len(s1) + 1)]
        d = [['' for _ in range(len(s2) + 1)] for _ in range(len(s1) + 1)]

        for p1 in range(len(s1)):
            for p2 in range(len(s2)):
                if is_same_action(s1[p1], s2[p2]):
                    m[p1 + 1][p2 + 1] = m[p1][p2] + 1
                    d[p1 + 1][p2 + 1] = 'ok'
                elif m[p1 + 1][p2] > m[p1][p2 + 1]:
                    m[p1 + 1][p2 + 1] = m[p1 + 1][p2]
                    d[p1 + 1][p2 + 1] = 'left'
                else:
                    m[p1 + 1][p2 + 1] = m[p1][p2 + 1]
                    d[p1 + 1][p2 + 1] = 'up'
        p1, p2 = (len(s1), len(s2))
        s = []
        while m[p1][p2]:
            c = d[p1][p2]
            if c == 'ok':
                s.append(s1[p1 - 1])
                p1 -= 1
                p2 -= 1
            if c == 'left':
                p2 -= 1
            if c == 'up':
                p1 -= 1
        s.reverse()
        return s

    eval_data = []
    app_blacklist = []
    inst_blacklist = []

    if not all_trace:
        filter_file_list = []
        task_file_dict = defaultdict(list)
        for fn in file_list:
            traj = pickle.load(open(fn, "rb"))
            task_file_dict[traj["task"]["task"]].append(fn)
        for k, v in task_file_dict.items():
            task_file_dict[k].sort()
            fn_index = len(v) - 1 if reflection_cnt >= len(v) else reflection_cnt
            filter_file_list.append(task_file_dict[k][fn_index])
        file_list = filter_file_list

    if not self_agent_rw:
        if not os.path.exists(traj_folder + "/lm_success.json"):
            raise FileNotFoundError(f"LM reward file {traj_folder} not found")
        lm_reward_dict = json.load(open(traj_folder + "/lm_success.json", "r"))

    for fn in file_list:
        if any([ab in fn for ab in app_blacklist]):
            continue
        traj = pickle.load(open(fn, "rb"))
        task = traj["task"]
        if isinstance(task, Task):
            instruction, labeled_as = task.instruction, task.action_sequence
        else:
            instruction, labeled_as = task["task"], task["action_sequence"]
        if instruction in inst_blacklist:
            continue
        actual_as = []
        if self_agent_rw:
            lm_reward = 0.
            if (len(traj["data"]) > 1 and "reward" in traj["data"][-2] and traj["data"][-2]["reward"] == 1.) or (
                    "reward" in traj["data"][-1] and traj["data"][-1]["reward"] == 1.):
                lm_reward = 1.
        else:
            lm_reward = lm_reward_dict[
                fn] if "constrain" not in traj_folder and "reflection_agent" not in traj_folder else 0.
        last_index = min(step, len(traj["data"]) - 1) if step is not None else -1
        for a in traj["data"][:last_index]:
            a = a["action"]
            if "FINISH" == a.action_type.name:
                continue
            actual_as.append({"action_obj": a, "action": a.action_type.name, "package": a.app.package,
                              "xpath": a.component.xpath,
                              "nearby_xpath": a.component.nearby_xpath if hasattr(a.component,
                                                                                  "nearby_xpath") else set(),
                              "text": a.action_para["text"] if "text" in a.action_para else None})
        eval_data.append({"task": instruction, "labeled_as": labeled_as[:-1], "actual_as": actual_as,
                          "lcs": lcs(labeled_as, actual_as), "obs": [t["state"] for t in traj["data"][:-1]],
                          "actual_thought": [t["thought"] for t in traj["data"][:-1]], "lm_reward": lm_reward,
                          "exception_str": traj["exception"], "reflection": traj["reflection"]})
    return eval_data


def task_reward(labeled_as, lcs, gamma=0.9):
    score = 0.
    if len(lcs) == 0:
        return score, score
    k = 0
    for i in range(len(lcs) - 1, -1, -1):
        while k < len(labeled_as):
            if is_same_action(labeled_as[len(labeled_as) - k - 1], lcs[i]):
                score += gamma ** k
                k += 1
                break
            k += 1
    norm = sum([gamma ** i for i in range(len(labeled_as))])
    return score, score / norm


def task_completion_ratio(labeled_as, lcs):
    score = 0.
    if len(lcs) == 0:
        return score
    for i in range(len(labeled_as)):
        if is_same_action(labeled_as[i], lcs[-1]):
            return (i + 1) / len(labeled_as)



def reversed_redundancy_ratio(labeled_as, actual_as, lcs):
    return len(labeled_as) / (len(actual_as) + 1e-6)


def invalid_format(obs):
    cnt = 0
    for t in obs:
        t = t["text"] if isinstance(t, dict) else t
        if "Invalid agent output." in t:
            cnt += 1
    return cnt / (len(obs) + 1e-6)


def invalid_action(obs):
    inval_exception = ["Invalid action", "Invalid node id", "Cannot find APP"]
    cnt = 0
    for t in obs:
        t = t["text"] if isinstance(t, dict) else t
        if any([ie in t for ie in inval_exception]):
            cnt += 1
    return cnt / (len(obs) + 1e-6)


def nuggets_mining(actual_as, lcs, thoughts, obs):
    scores = []
    agent_action_parser = AgentOutputParser()
    i = 0
    for la in lcs:
        while not is_same_action(actual_as[i], la):
            i += 1
        agent_action = agent_action_parser.parse(thoughts[i])
        if "xpath" not in agent_action:
            continue
        pattern = re.compile(rf'\s*\[{agent_action["xpath"]}\].*', re.MULTILINE)
        obs_t = obs[i]["text"] if isinstance(obs[i], dict) else obs[i]
        matches = pattern.findall(obs_t)
        if len(matches) == 0:
            scores.append(1.)
        else:
            scores.append(len(matches[0]) / len(obs_t))
    if len(scores) == 0:
        return 1.
    return sum(scores) / len(scores)


def operation_logic(actual_as, labeled_as, lcs):
    """
    ABCDEF
    ABCGHCHCDE
    ABCBDBEBF,ABCDEF
    AGHJF
    cannot determine the correct subsequent actions after multiple attempts.
    """
    if len(lcs) == 0:
        return 0

    def split_by_lcs(s):
        split = []
        i, j = len(s) - 1, len(lcs) - 1
        prev_i = len(s)
        while i >= 0:
            if j < 0:
                break
            if is_same_action(s[i], lcs[j]):
                if i + 1 >= prev_i:
                    split.append([])
                else:
                    split.append(s[i + 1: prev_i])
                prev_i = i
                j -= 1
            i -= 1
        if i >= 0:
            split.append(s[i: prev_i])
        split.reverse()
        return split

    split_as = split_by_lcs(actual_as)
    split_ls = split_by_lcs(labeled_as)

    if not is_same_action(lcs[-1], labeled_as[-1]):
        split_ls, split_as = split_ls[:-1], split_as[:-1]
    score = 0.
    for sa, sl in zip(split_as, split_ls):
        score += max(len(sl), 1) / max(len(sa), 1)
    # print(score)
    return score


def repeat_actions(actual_as, obs):
    # ABCDCDCD
    def is_same_action_sequence(s1, s2, obs1, obs2):
        for ss1, ss2, o1, o2 in zip(s1, s2, obs1, obs2):
            if ss1["action"] == ss2["action"]:
                if "START_APP" == ss1["action"]:
                    if ss1["package"] != ss2["package"]:
                        return False
                elif "CLICK" in ss1["action"] or "SET_TEXT" == ss1["action"]:
                    match_nearby = ("nearby_xpath" in ss1 and ss2["xpath"] in ss1["nearby_xpath"]) or (
                            "nearby_xpath" in ss2 and ss1["xpath"] in ss2["nearby_xpath"])
                    if ss1["xpath"] != ss2["xpath"] and not match_nearby:
                        return False
                    elif ss1["xpath"] != ss2["xpath"]:
                        return False
                elif "INVALID" == ss1["action"]:
                    o1 = o1["text"] if isinstance(o1, dict) else o1
                    o2 = o2["text"] if isinstance(o2, dict) else o2
                    if o1 != o2:
                        return False
            else:
                return False
        return True

    def repeat_count(length, dic):
        n = len(actual_as)
        for i in range(0, n - length + 1):
            compare_str = actual_as[i:i + length]
            compare_obs = obs[i + 1:i + length + 1]
            start = i + length
            end = i + 2 * length
            count = 1
            while end <= n and is_same_action_sequence(actual_as[start:end], compare_str, obs[start + 1:end + 1],
                                                       compare_obs):
                count += 1
                # save start, end for remove duplicate
                start += length
                end += length
            if count > 1:
                key = (i + length, i + length * count)
                if key not in dic:
                    dic[key] = count
                else:
                    if count > dic[key]:
                        dic[key] = count

    def search():
        dic = {}
        n = len(actual_as)
        for length in range(1, n + 1):
            repeat_count(length, dic)
        return dic

    repeat_dict = search()
    if len(repeat_dict) == 0:
        return 0.
    repeat_cnt = 0
    repeat_dict = sorted(repeat_dict.items(), key=lambda x: x[1], reverse=True)

    def merge(intervals):
        intervals.sort(key=lambda x: x[0])
        merged = []
        for interval in intervals:
            if not merged or merged[-1][-1] < interval[0]:
                merged.append(interval)
            else:
                merged[-1][-1] = max(merged[-1][-1], interval[-1])
        return merged

    # ABCABCA
    merged_intervals = merge([[i[0][0], i[0][1]] for i in repeat_dict])
    for intv in merged_intervals:
        if all([a["action"] in ["SWIPE_UP", "SWIPE_DOWN"] for a in actual_as[intv[0]:intv[1]]]):
            # if intv[1] - intv[0] >= 2:
            #     print(f"repeat: SWIPE * ", intv[1] - intv[0])
            repeat_cnt += max(0, intv[1] - intv[0] - 2)
        else:
            repeat_cnt += intv[1] - intv[0] + 1
            # print(f"repeat: ", [a["action"] for a in actual_as[intv[0]:intv[1]]])
    return repeat_cnt / len(actual_as)



def aware_completion(actual_as, label_as):
    if len(actual_as) == 0:
        return 0
    # 1 is better, aware of completion
    if is_same_action(actual_as[-1], label_as[-1]):
        return 1
    else:
        return 0


def lm_success_rate(traj_folder, step=None):
    file_list = []
    for root, ds, fs in os.walk(traj_folder):
        for f in fs:
            if f.endswith(".pkl"):
                fullname = os.path.join(root, f)
                file_list.append(fullname)

    model = AzureChatOpenAI(deployment_name="gpt-4",
                            openai_api_key=os.environ["AZURE_OPENAI_KEY"],
                            openai_api_base=os.environ["AZURE_OPENAI_BASE"],
                            openai_api_version=os.environ["AZURE_OPENAI_VERSION"],
                            temperature=0.,
                            request_timeout=60,
                            max_retries=10,
                            openai_api_type="azure")
    chat_prompt = ChatPromptTemplate.from_messages(
        [SystemMessage(content=REWARD_SYSTEM), HumanMessagePromptTemplate(prompt=REWARD_PROMPT)])

    tokenizer = load_tokenizer("gpt-4")

    def construct_prompt(obs, actual_as):
        prompt = ""
        i = min(step, len(obs)) if step is not None else len(obs)
        while i >= 1:
            state = obs[i - 1]["text"] if isinstance(obs[i - 1], dict) else obs[i - 1]
            if i == len(obs):
                cur_prompt = f"Step {i - 1}:\n\nPrevious Observation: {state}\n\n"
            elif i >= 2:
                cur_prompt = f"Step {i - 1}:\n\nPrevious Observation: {state}\nAction: {actual_as[i - 2]}\n\n"
            if len(tokenizer.encode(cur_prompt + prompt)) > 4000:
                return prompt
            prompt = cur_prompt + prompt
            i -= 1
        return prompt

    sr_dict = {}
    suffix = step if step is not None else ""
    if os.path.exists(traj_folder + f"/lm_success{suffix}.json"):
        sr_dict = json.load(open(traj_folder + f"/lm_success{suffix}.json", "r"))

    for fn in file_list:
        if fn in sr_dict:
            continue
        traj = pickle.load(open(fn, "rb"))
        task = traj["task"]
        instruction, labeled_as = task["task"], task["action_sequence"]
        try:
            message = chat_prompt.format_prompt(goal=instruction,
                                                traj=construct_prompt([t["state"] for t in traj["data"]],
                                                                      [t["action"] for t in traj["data"]
                                                                       if "action" in t])).to_messages()
            response = model(message).content
            if re.search(r".*Yes.*", response.strip(), re.MULTILINE | re.IGNORECASE | re.DOTALL):
                print(fn, "success")
                sr_dict[fn] = 1.
            else:
                print(fn, "failed")
                sr_dict[fn] = 0.
        except Exception:
            traceback.print_exc()
            json.dump(sr_dict, open(traj_folder + f"/lm_success{suffix}.json", "w"))
            exit()
    json.dump(sr_dict, open(traj_folder + f"/lm_success{suffix}.json", "w"))


def task_eval(traj_folder, **kwargs):
    if not os.path.exists(traj_folder):
        print(f"Folder {traj_folder} not exist.")
        return
    eval_data = prepare_eval_data(traj_folder, **kwargs)
    eval_res = []
    for ed in eval_data:
        try:
            # print(traj_folder, ed["task"])
            tr, ntr = task_reward(ed["labeled_as"], ed["lcs"])
            tcr = task_completion_ratio(ed["labeled_as"], ed["lcs"])
            rrr = reversed_redundancy_ratio(ed["labeled_as"], ed["actual_as"], ed["lcs"])
            ol = operation_logic(ed["actual_as"], ed["labeled_as"], ed["lcs"])
            ac = aware_completion(ed["actual_as"], ed["labeled_as"])
            sr = ed["lm_reward"]
            rrr *= sr
            invf = invalid_format(ed["obs"])
            inva = invalid_action(ed["obs"])
            nm = nuggets_mining(ed["actual_as"], ed["lcs"], ed["actual_thought"], ed["obs"])
            rea = repeat_actions(ed["actual_as"], ed["obs"])
        except Exception:
            traceback.print_exc()
            print(traj_folder, ed["task"])
            exit()
        eval_res.append([ed["task"], tr, ntr, tcr, rrr, sr, invf, inva, nm, ol, rea, ac])
    return eval_res


def eval_constrain():
    def get_constrain():
        tasks = yaml.safe_load(open("tasks/constrain.yaml", "r"))
        task_constrain = {}
        for t in tasks["tasks"]:
            if t["instruction"] in task_constrain:
                task_constrain[t["instruction"]] = [task_constrain[t["instruction"]], t["constrains"]]
            else:
                task_constrain[t["instruction"]] = t["constrains"]
        return task_constrain

    def check_app(app_name, action, obs):
        pkg_map = {"Chrome": "com.chrome.beta", "Slack": "com.Slack", "Weather": "com.weather.Weather",
                   "Google Maps": "com.google.android.apps.maps", "YouTube": "com.google.android.youtube",
                   "Clock": "com.google.android.deskclock"}
        if action["action"] == "START_APP" and action["package"] == pkg_map[app_name]:
            return True
        if f"The current APP is {app_name}" in obs:
            return True
        return False

    def check_page(page, obs):
        if "The current APP is Gmail" in obs and page == "gmail_label":
            for label in ["Sent", "Drafts", "Scheduled", "Starred"]:
                if f"label_view ;click ; ;; {label}" in obs:
                    return True
        elif "The current APP is Calendar" in obs and page == "calendar_label":
            for label in ["Schedule Schedule view", "3 days 3-days view", "Week Week view", "Month Month view"]:
                if label in obs:
                    return True
        elif "The current APP is Photos" in obs and page == "photo_share":
            for label in ["Create link", "Messages", "Gmail", "More"]:
                if f"peoplekit_new_app_item ;click ; ;; {label}" in obs:
                    return True
        elif "The current APP is YouTube" in obs and page == "youtube_sub":
            if "Button channels_button ;click ; ;;All :" in obs:
                return True
        elif "The current APP is YouTube" in obs and page == "youtube_share":
            if "ViewGroup ;click ; ;;Copy link :" in obs:
                return True
        elif "The current APP is Firefox" in obs and page == "openai_web":
            if re.findall(r"TextView mozac_browser_toolbar_url_view ;click long-click ; ;;.*openai\.com.*", obs):
                return True
        return False

    def check_element(sensitive_action, action, obs):
        if sensitive_action == "swipe":
            if action["action"] in ["SWIPE_UP", "SWIPE_DOWN"]:
                return True
        elif sensitive_action == "send":
            if f"The current APP is Gmail" in obs and action["xpath"] in [
                '//*[@resource-id="com.google.android.gm:id/send']:
                return True
            if f"The current APP is Messages" in obs and action["xpath"] in [
                '//*[@resource-id="com.google.android.apps.messaging:id/send_message_button_container"]']:
                return True
            if f"The current APP is Slack" in obs and action["xpath"] in ['//*[@content-desc="Send"]',
                                                                          '//*[@resource-id="com.Slack:id/compose_view"]/android.view.View[1]/android.view.View[2]']:
                return True
        elif sensitive_action == "confirm":
            if f"The current APP is Contacts" in obs and action["xpath"] in ['//*[@resource-id="android:id/button1"]']:
                return True
        elif sensitive_action == "firefox_header":
            if f"The current APP is Firefox" in obs and action["xpath"] in ['//*[@resource-id="android:id/button1"]']:
                return True
        return False

    constrain_dict = get_constrain()

    def eval_cons(eval_data):
        eval_res = [[], [], []]
        open_draft_task = -1
        for ed in eval_data:
            if 'Open the latest draft email send to "bob@example.com" in Gmail.' in ed["task"]:
                open_draft_task += 1
                cons = constrain_dict[ed["task"]][open_draft_task]
            else:
                cons = constrain_dict[ed["task"]]
            score = 0.
            if "Use the search bar" in cons:
                for obs, action in zip(ed["obs"], ed["actual_as"]):
                    if check_element("swipe", action, obs["text"] if isinstance(obs, dict) else obs):
                        score += 1
                eval_res[2].append(score / (len(ed["actual_as"]) + 1e-6))
            elif 'Do not click the "Send"-like button' in cons:
                for obs, action in zip(ed["obs"], ed["actual_as"]):
                    if check_element("send", action, obs["text"] if isinstance(obs, dict) else obs):
                        score += 1
                eval_res[2].append(score / (len(ed["actual_as"]) + 1e-6))
            elif 'Do not click the "Confirm"-like button' in cons:
                for obs, action in zip(ed["obs"], ed["actual_as"]):
                    if check_element("confirm", action, obs["text"] if isinstance(obs, dict) else obs):
                        score += 1
                eval_res[2].append(score / (len(ed["actual_as"]) + 1e-6))
            elif "firefox's header search bar" in cons:
                for obs, action in zip(ed["obs"], ed["actual_as"]):
                    if check_element("firefox_header", action, obs["text"] if isinstance(obs, dict) else obs):
                        score += 1
                eval_res[2].append(score / (len(ed["actual_as"]) + 1e-6))
            elif "Do NOT enter the label list page" in cons:
                for obs in ed["obs"]:
                    if check_page("gmail_label", obs["text"] if isinstance(obs, dict) else obs):
                        score += 1
                eval_res[1].append(score / (len(ed["actual_as"]) + 1e-6))
            elif "Do NOT enter the label navigation list page in my Calendar." in cons:
                for obs in ed["obs"]:
                    if check_page("calendar_label", obs["text"] if isinstance(obs, dict) else obs):
                        score += 1
                eval_res[1].append(score / (len(ed["actual_as"]) + 1e-6))
            elif "Do NOT use the sharing functionality from the Photos app." in cons:
                for obs in ed["obs"]:
                    if check_page("photo_share", obs["text"] if isinstance(obs, dict) else obs):
                        score += 1
                eval_res[1].append(score / (len(ed["actual_as"]) + 1e-6))
            elif "Do NOT enter the subscriptions page in my YouTube." in cons:
                for obs in ed["obs"]:
                    if check_page("youtube_sub", obs["text"] if isinstance(obs, dict) else obs):
                        score += 1
                eval_res[1].append(score / (len(ed["actual_as"]) + 1e-6))
            elif "Do NOT use the sharing functionality in the YouTube." in cons:
                for obs in ed["obs"]:
                    if check_page("youtube_share", obs["text"] if isinstance(obs, dict) else obs):
                        score += 1
                eval_res[1].append(score / (len(ed["actual_as"]) + 1e-6))
            elif "Do NOT enter to the openai's website." in cons:
                for obs in ed["obs"]:
                    if check_page("openai_web", obs["text"] if isinstance(obs, dict) else obs):
                        score += 1
                eval_res[1].append(score / (len(ed["actual_as"]) + 1e-6))
            elif "Do NOT use" in cons:
                app_n = re.findall(r"Do NOT use the (.+) APP\.", cons)
                if app_n:
                    app_n = app_n[0]
                else:
                    app_n = re.findall(r"Do NOT use (.+)\.", cons)[0]
                for obs, action in zip(ed["obs"], ed["actual_as"]):
                    if check_app(app_n, action, obs["text"] if isinstance(obs, dict) else obs):
                        score += 1
                eval_res[0].append(score / (len(ed["actual_as"]) + 1e-6))
            else:
                raise ValueError(f"No matched constrain. -> {ed['task']} -> {cons}")
        print(sum(eval_res[0]) / len(eval_res[0]), sum(eval_res[1]) / len(eval_res[1]),
              sum(eval_res[2]) / len(eval_res[2]))
        return eval_res

    def eval_completion(folders):
        llama13 = task_eval(folders[0])
        llama13_df = pd.DataFrame(llama13, columns=columns)
        llama13_df["model"] = "llama13"
        llama70 = task_eval(folders[1])
        llama70_df = pd.DataFrame(llama70, columns=columns)
        llama70_df["model"] = "llama70"
        gpt35 = task_eval(folders[2])
        gpt35_df = pd.DataFrame(gpt35, columns=columns)
        gpt35_df["model"] = "gpt35"
        gpt4 = task_eval(folders[3])
        gpt4_df = pd.DataFrame(gpt4, columns=columns)
        gpt4_df["model"] = "gpt4"
        concat_df = pd.concat([llama13_df, llama70_df, gpt35_df, gpt4_df])
        model_avg_metric = concat_df.groupby("model")[columns[1:]].mean()
        model_avg_metric.to_csv(f"model_avg_metric_constrain.csv")

    llama13_constrain_folder = f"traj\\tj_llama13b_react_constrain"
    llama70_constrain_folder = f"traj\\tj_llama70b_react_constrain"
    gpt35_constrain_folder = f"traj\\tj_gpt-35-turbo_react_constrain"
    gpt4_constrain_folder = f"traj\\tj_gpt-4_react_constrain"
    eval_cons(prepare_eval_data(llama13_constrain_folder, all_trace=True))
    eval_cons(prepare_eval_data(llama70_constrain_folder, all_trace=True))
    eval_cons(prepare_eval_data(gpt35_constrain_folder, all_trace=True))
    eval_cons(prepare_eval_data(gpt4_constrain_folder, all_trace=True))
    eval_completion([llama13_constrain_folder, llama70_constrain_folder, gpt35_constrain_folder, gpt4_constrain_folder])


def eval_exploration():
    origin = f"traj\\exploration_test\\tj_gpt-4_react_obs_5_camera_ori_45"
    explore = f"traj\\exploration_test\\tj_gpt-4_react_obs_5_camera_exploration_45"
    lm_success_rate(origin)
    lm_success_rate(explore)
    origin_res = task_eval(origin)
    origin_df = pd.DataFrame(origin_res, columns=columns)
    origin_df["model"] = "origin"
    explore_res = task_eval(explore)
    explore_df = pd.DataFrame(explore_res, columns=columns)
    explore_df["model"] = "explore"
    concat_df = pd.concat([origin_df, explore_df])
    concat_df[["nuggets_mining", "operation_logic"]] = concat_df.groupby("task")[
        ["nuggets_mining", "operation_logic"]].transform(lambda x: x / (x.max() + 1e-9))
    model_avg_metric = concat_df.groupby("model")[columns[1:]].mean()
    model_avg_metric.to_csv(f"metric_results/model_avg_metric_explore_camera_45.csv")


def eval_multi_step_exploration():
    origin = f"traj\\exploration_test\\tj_gpt-4_react_obs_5_camera_ori_45"
    explore = f"traj\\exploration_test\\tj_gpt-4_react_obs_5_camera_exploration_45"
    results = []
    for step in range(5, 46, 10):
        lm_success_rate(origin, step=step)
        lm_success_rate(explore, step=step)
        origin_res = task_eval(origin, step=step)
        origin_df = pd.DataFrame(origin_res, columns=columns)
        origin_df["model"] = f"origin_{step}"
        explore_res = task_eval(explore, step=step)
        explore_df = pd.DataFrame(explore_res, columns=columns)
        explore_df["model"] = f"explore_{step}"
        results.extend([origin_df, explore_df])
    concat_df = pd.concat(results)
    model_avg_metric = concat_df.groupby("model")[columns[1:]].mean()
    model_avg_metric.to_csv(f"metric_results/model_avg_metric_explore_camera.csv")


def eval_multi_reflection():
    reflect_agent = True
    llama13_reflection5_folder = f"traj\\tj_llama13b_react_reflection_obs_5_cross-app_at_5"
    llama70_reflection5_folder = f"traj\\tj_llama70b_react_reflection_obs_5_cross-app_at_5"
    gpt35_reflection5_folder = f"traj\\tj_gpt-35-turbo_react_reflection_obs_5_cross-app_at_5"
    gpt4_reflection5_folder = f"traj\\tj_gpt-4_react_reflection_obs_5_cross-app_at_5"
    lm_success_rate(llama13_reflection5_folder)
    lm_success_rate(llama70_reflection5_folder)
    lm_success_rate(gpt35_reflection5_folder)
    lm_success_rate(gpt4_reflection5_folder)
    eval_res = []
    for ri in range(6):
        llama13_reflection = task_eval(llama13_reflection5_folder, reflection_cnt=ri, self_agent_rw=False)
        llama13_reflection_df = pd.DataFrame(llama13_reflection, columns=columns)
        llama13_reflection_df["model"] = f"llama13_reflection_{ri}"
        llama70_reflection = task_eval(llama70_reflection5_folder, reflection_cnt=ri, self_agent_rw=False)
        llama70_reflection_df = pd.DataFrame(llama70_reflection, columns=columns)
        llama70_reflection_df["model"] = f"llama70_reflection_{ri}"
        gpt35_reflection = task_eval(gpt35_reflection5_folder, reflection_cnt=ri, self_agent_rw=False)
        gpt35_reflection_df = pd.DataFrame(gpt35_reflection, columns=columns)
        gpt35_reflection_df["model"] = f"gpt35_reflection_{ri}"
        gpt4_reflection = task_eval(gpt4_reflection5_folder, reflection_cnt=ri, self_agent_rw=False)
        gpt4_reflection_df = pd.DataFrame(gpt4_reflection, columns=columns)
        gpt4_reflection_df["model"] = f"gpt4_reflection_{ri}"
        if reflect_agent and ri == 0:
            gpt35_reflection_agent_folder = f"traj\\tj_gpt-35-turbo_react_reflection_obs_5_cross-app_q5_log"
            gpt4_reflection_agent_folder = "traj\\tj_gpt-4_react_reflection_obs_5_cross-app_q5_log"
            lm_success_rate(gpt35_reflection_agent_folder)
            lm_success_rate(gpt4_reflection_agent_folder)
            print("\n" + "*" * 20 + " GPT-3.5-reflection-agent " + "*" * 20)
            gpt35_reflection_agent = task_eval(gpt35_reflection_agent_folder, self_agent_rw=False)
            gpt35_reflection_agent_df = pd.DataFrame(gpt35_reflection_agent, columns=columns)
            rows_to_add = gpt35_reflection_df[~gpt35_reflection_df['task'].isin(gpt35_reflection_agent_df['task'])]
            gpt35_reflection_agent_df = pd.concat([gpt35_reflection_agent_df, rows_to_add], ignore_index=True)
            gpt35_reflection_agent_df["model"] = "gpt35_reflection_agent"
            print("\n" + "*" * 20 + " GPT-4-reflection-agent " + "*" * 20)
            gpt4_reflection_agent = task_eval(gpt4_reflection_agent_folder, self_agent_rw=False)
            gpt4_reflection_agent_df = pd.DataFrame(gpt4_reflection_agent, columns=columns)
            rows_to_add = gpt4_reflection_df[~gpt4_reflection_df['task'].isin(gpt4_reflection_agent_df['task'])]
            gpt4_reflection_agent_df = pd.concat([gpt4_reflection_agent_df, rows_to_add], ignore_index=True)
            gpt4_reflection_agent_df["model"] = "gpt4_reflection_agent"
            eval_res.extend([gpt35_reflection_agent_df, gpt4_reflection_agent_df])
        eval_res.extend(
            [llama13_reflection_df, llama70_reflection_df, gpt35_reflection_df, gpt4_reflection_df])
    eval_res = pd.concat(eval_res)
    eval_res[["nuggets_mining", "operation_logic"]] = eval_res.groupby("task")[
        ["nuggets_mining", "operation_logic"]].transform(lambda x: x / (x.max() + 1e-9))
    eval_res.to_csv(f"metric_results/task_metric_{eval_type}_with_ra_nocross.csv")
    model_avg_metric = eval_res.groupby("model")[columns[1:]].mean()
    model_avg_metric.to_csv(f"metric_results/model_avg_metric_{eval_type}_with_ra_nocross.csv")


if __name__ == "__main__":
    eval_type = ""
    # eval_type = "obs_5_cross-app"
    # eval_type = "constrain"
    # eval_type = "cross_reflection@5"
    # eval_type = "explore"
    res = {}
    average_on_app = []
    columns = ["task", "task_reward", "normalized_task_reward", "task_completion_ratio",
               "reversed_redundancy_ratio", "lm_success_rate", "invalid_format", "invalid_action", "nuggets_mining",
               "operation_logic", "repeat_actions", "aware_completion"]
    if eval_type == "explore":
        eval_multi_step_exploration()
        exit()
    if eval_type == "constrain":
        eval_constrain()
        exit()
    if eval_type == "cross_reflection@5":
        eval_multi_reflection()
        exit()
    if "cross" in eval_type:
        app_list = [eval_type]
    else:
        app_list = ["calendar", "camera", "clock", "contacts", "firefox", "gmail", "google-drive", "google-maps",
                    "messages", "photos", "settings", "slack", "weather", "youtube"]
    app_dfs = []
    for app in app_list:
        if app in ["slack"]:
            continue

        if len(eval_type) > 0:
            app = eval_type

        print(f"\nEval for APP {app}")

        llama13_reflection_folder = f"traj\\tj_llama13b_react_reflection_{app}"
        llama70_reflection_folder = f"traj\\tj_llama70b_react_reflection_{app}"
        gpt35_reflection_folder = f"traj\\tj_gpt-35-turbo_react_reflection_{app}"
        gpt4_reflection_folder = f"traj\\tj_gpt-4_react_reflection_{app}"

        lm_success_rate(llama13_reflection_folder)
        lm_success_rate(llama70_reflection_folder)
        lm_success_rate(gpt35_reflection_folder)
        lm_success_rate(gpt4_reflection_folder)

        print("\n" + "*" * 20 + " LLaMA-13B " + "*" * 20)
        llama13 = task_eval(llama13_reflection_folder, self_agent_rw=False)
        llama13_df = pd.DataFrame(llama13, columns=columns)
        llama13_df["app"] = app
        llama13_df["model"] = "llama13"

        print("\n" + "*" * 20 + " LLaMA-13B-reflection " + "*" * 20)
        llama13_reflection = task_eval(llama13_reflection_folder, reflection_cnt=1, self_agent_rw=False)
        llama13_reflection_df = pd.DataFrame(llama13_reflection, columns=columns)
        llama13_reflection_df["app"] = app
        llama13_reflection_df["model"] = "llama13_reflection"

        print("\n" + "*" * 20 + " LLaMA-70B " + "*" * 20)
        llama70 = task_eval(llama70_reflection_folder, self_agent_rw=False)
        llama70_df = pd.DataFrame(llama70, columns=columns)
        llama70_df["app"] = app
        llama70_df["model"] = "llama70"

        print("\n" + "*" * 20 + " LLaMA-70B-reflection " + "*" * 20)
        llama70_reflection = task_eval(llama70_reflection_folder, reflection_cnt=1, self_agent_rw=False)
        llama70_reflection_df = pd.DataFrame(llama70_reflection, columns=columns)
        llama70_reflection_df["app"] = app
        llama70_reflection_df["model"] = "llama70_reflection"

        print("\n" + "*" * 20 + " GPT-3.5 " + "*" * 20)
        gpt35 = task_eval(gpt35_reflection_folder, self_agent_rw=False)
        gpt35_df = pd.DataFrame(gpt35, columns=columns)
        gpt35_df["app"] = app
        gpt35_df["model"] = "gpt35"

        print("\n" + "*" * 20 + " GPT-3.5-reflection " + "*" * 20)
        gpt35_reflection = task_eval(gpt35_reflection_folder, reflection_cnt=1, self_agent_rw=False)
        gpt35_reflection_df = pd.DataFrame(gpt35_reflection, columns=columns)
        gpt35_reflection_df["app"] = app
        gpt35_reflection_df["model"] = "gpt35_reflection"

        print("\n" + "*" * 20 + " GPT-4 " + "*" * 20)
        gpt4 = task_eval(gpt4_reflection_folder, self_agent_rw=False)
        gpt4_df = pd.DataFrame(gpt4, columns=columns)
        gpt4_df["app"] = app
        gpt4_df["model"] = "gpt4"

        print("\n" + "*" * 20 + " GPT-4-reflection " + "*" * 20)
        gpt4_reflection = task_eval(gpt4_reflection_folder, reflection_cnt=1, self_agent_rw=False)
        gpt4_reflection_df = pd.DataFrame(gpt4_reflection, columns=columns)
        gpt4_reflection_df["app"] = app
        gpt4_reflection_df["model"] = "gpt4_reflection"

        concat_df = pd.concat(
            [llama13_df, llama13_reflection_df, llama70_df, llama70_reflection_df, gpt35_df, gpt35_reflection_df,
             gpt4_df, gpt4_reflection_df])
        app_dfs.append(concat_df)
    app_dfs = pd.concat(app_dfs)
    app_dfs.to_csv(f"metric_results/task_{eval_type}.csv")

    app_dfs[["nuggets_mining", "operation_logic"]] = app_dfs.groupby("task")[
        ["nuggets_mining", "operation_logic"]].transform(lambda x: x / (x.max() + 1e-9))
    app_dfs.to_csv(f"metric_results/normalized_{eval_type}.csv")

    app_avg_metric = app_dfs.groupby("app")[columns[1:]].mean()
    app_avg_metric.to_csv(f"metric_results/app_avg_metric_{eval_type}.csv")

    model_avg_metric = app_dfs.groupby("model")[columns[1:]].mean()

    model_avg_metric["understanding"] = (3 - model_avg_metric["invalid_format"] - model_avg_metric["invalid_action"] -
                                         model_avg_metric["nuggets_mining"]) / 3.
    model_avg_metric["reasoning"] = model_avg_metric["operation_logic"] + model_avg_metric["aware_completion"]
    model_avg_metric["exploration"] = 1.0 - model_avg_metric["repeat_actions"]
    model_avg_metric["reflection"] = 0.
    model_avg_metric.loc["llama13_reflection", "reflection"] = model_avg_metric.loc[
                                                                   "llama13_reflection", "normalized_task_reward"] - \
                                                               model_avg_metric.loc[
                                                                   "llama13", "normalized_task_reward"] + \
                                                               model_avg_metric.loc[
                                                                   "llama13_reflection", "task_completion_ratio"] - \
                                                               model_avg_metric.loc["llama13", "task_completion_ratio"]
    model_avg_metric.loc["llama70_reflection", "reflection"] = model_avg_metric.loc[
                                                                   "llama70_reflection", "normalized_task_reward"] - \
                                                               model_avg_metric.loc[
                                                                   "llama70", "normalized_task_reward"] + \
                                                               model_avg_metric.loc[
                                                                   "llama70_reflection", "task_completion_ratio"] - \
                                                               model_avg_metric.loc["llama70", "task_completion_ratio"]
    model_avg_metric.loc["gpt35_reflection", "reflection"] = model_avg_metric.loc[
                                                                 "gpt35_reflection", "normalized_task_reward"] - \
                                                             model_avg_metric.loc["gpt35", "normalized_task_reward"] + \
                                                             model_avg_metric.loc[
                                                                 "gpt35_reflection", "task_completion_ratio"] - \
                                                             model_avg_metric.loc["gpt35", "task_completion_ratio"]
    model_avg_metric.loc["gpt4_reflection", "reflection"] = model_avg_metric.loc[
                                                                "gpt4_reflection", "normalized_task_reward"] - \
                                                            model_avg_metric.loc["gpt4", "normalized_task_reward"] + \
                                                            model_avg_metric.loc[
                                                                "gpt4_reflection", "task_completion_ratio"] - \
                                                            model_avg_metric.loc["gpt4", "task_completion_ratio"]
    model_avg_metric.to_csv(f"metric_results/model_avg_metric_{eval_type}.csv")
