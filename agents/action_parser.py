import re

from android_env2.exception import OutputParserException


class AgentOutputParser:
    def __init__(self):
        self.action_splitter = "#"
        self.arg_splitter = ["\\[", "\\]"]

    def parse_arg(self, arg):
        pattern = rf"{self.arg_splitter[0]}(.+?){self.arg_splitter[1]}"
        match = re.findall(pattern, arg)
        if len(match) > 1:
            raise OutputParserException("Invalid agent output. Only one action output is allowed.")
        if match:
            para = match[-1]
            return para
        else:
            raise OutputParserException("Invalid agent output. At least output one action.")

    def parse(self, response):
        pattern = rf"{self.action_splitter}(.+?){self.action_splitter}"
        match = re.findall(pattern, response)
        if match:
            action = match[-1]
        else:
            action = response
        action = action.split()
        if "start" in action[0]:
            return {"action": "START_APP", "package": self.parse_arg(" ".join(action[1:]))}
        elif "stop" in action[0]:
            return {"action": "STOP_APP", "package": self.parse_arg(" ".join(action[1:]))}
        elif "long_click" in action[0]:
            return {"action": "LONG_CLICK", "xpath": self.parse_arg(" ".join(action[1:]))}
        elif "click" in action[0]:
            return {"action": "CLICK", "xpath": self.parse_arg(" ".join(action[1:]))}
        elif "set_text" in action[0]:
            return {"action": "SET_TEXT", "xpath": self.parse_arg(action[1]),
                    "text": self.parse_arg(" ".join(action[2:]))}
        elif action[0] in ["swipe_up", "scroll_down", "swipe_down", "swipe_left", "swipe_right", "press_back",
                           "press_recent", "press_enter"]:
            if action[0] == "scroll_down":
                action[0] = "swipe_up"
            return {"action": action[0].upper()}
        elif "finish" in action[0]:
            response = ""
            if len(action) > 1:
                response = " ".join(action[1:])
            return {"action": "FINISH", "text": response}
        else:
            raise OutputParserException(f"Invalid action: {action}")


class RegexParser(AgentOutputParser):
    pass


class LLMParser(AgentOutputParser):
    pass
