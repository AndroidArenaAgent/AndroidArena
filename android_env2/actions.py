import re
from enum import Enum
from typing import Dict

import numpy as np
from gymnasium import spaces, ActionWrapper
from gymnasium.core import WrapperActType, ActType

from android_env2.exception import AndroidActionException
from android_env2.phone import APP, Activity, Component
from android_env2.constant import TEXT_MAX_LENGTH


class ActionType(Enum):
    NONE = 0
    # app level
    INSTALL_APP = 1
    START_APP = 2
    STOP_APP = 3
    STOP_ALL_APP = 4

    # component level
    CLICK = 5
    DOUBLE_CLICK = 6
    LONG_CLICK = 7
    SET_TEXT = 8

    # system level
    PRESS_BACK = 9
    PRESS_HOME = 10
    SCREEN_ON = 11
    SCREEN_OFF = 12
    VOLUME_UP = 13
    VOLUME_DOWN = 14
    VOLUME_MUTE = 15
    SET_ORIENTATION = 16
    FREEZE_ROTATION = 17
    UNFREEZE_ROTATION = 18
    SCREENSHOT = 19
    SWIPE_UP = 20
    SWIPE_DOWN = 21
    SWIPE_LEFT = 22
    SWIPE_RIGHT = 23
    SWIPE = 24
    RECENT = 25
    DRAG = 27
    LIST_ALL_APP = 28
    PRESS_ENTER = 29

    FINISH = 30
    INVALID = 31

    def __str__(self) -> str:
        return f"ActionType.{self.name}"


class Action:
    def __init__(self):
        self.action_type: ActionType = ActionType.NONE
        self.action_para: Dict[str, str] = dict()
        self.app: APP = APP()
        self.activity: Activity = Activity()
        self.component: Component = Component()

    def __str__(self):
        match self.action_type:
            case ActionType.INSTALL_APP:
                return f"install {self.app.name} APP"
            case ActionType.START_APP:
                return f"launch {self.app.name} APP"
            case ActionType.CLICK:
                return f"click {self.component.name} on {self.app.name} APP"
            case ActionType.SET_TEXT:
                return f"type {self.action_para['text']} in {self.component.name} of {self.app.name} APP"
            case ActionType.STOP_APP:
                return f"stop {self.app.name} APP"
            case ActionType.STOP_ALL_APP:
                return "stop all APPs"
            case ActionType.DOUBLE_CLICK:
                return f"double click {self.component.name} on {self.app.name} APP"
            case ActionType.LONG_CLICK:
                return f"long click {self.component.name} on {self.app.name} APP"
            case ActionType.PRESS_BACK:
                return "press the back key"
            case ActionType.PRESS_HOME:
                return "press the home key"
            case ActionType.PRESS_ENTER:
                return "press the enter key"
            case ActionType.SCREEN_ON:
                return "turn on the screen"
            case ActionType.SCREEN_OFF:
                return "turn off the screen"
            case ActionType.VOLUME_UP:
                return "turn the volume up"
            case ActionType.VOLUME_DOWN:
                return "turn the volume down"
            case ActionType.VOLUME_MUTE:
                return "mute the volume"
            case ActionType.SET_ORIENTATION:
                return f"rotate screen to {self.action_para['orientation']}"
            case ActionType.FREEZE_ROTATION:
                return "freeze screen rotation"
            case ActionType.UNFREEZE_ROTATION:
                return "un-freeze screen rotation"
            case ActionType.SCREENSHOT:
                return "take a screenshot"
            case ActionType.SWIPE_UP:
                return f"swip up on {self.app.name} APP"
            case ActionType.SWIPE_DOWN:
                return f"swip down on {self.app.name} APP"
            case ActionType.SWIPE_LEFT:
                return f"swip left on {self.app.name} APP"
            case ActionType.SWIPE_RIGHT:
                return f"swip right on {self.app.name} APP"
            case ActionType.SWIPE:
                return (f"swipe from [{self.action_para['sx']}, {self.action_para['sy']}]  "
                        f"to [{self.action_para['ex']}, {self.action_para['ey']}] on {self.app.name} APP")
            case ActionType.RECENT:
                return "show recent Apps"
            case ActionType.DRAG:
                return (f"drag from [{self.action_para['sx']}, {self.action_para['sy']}]  "
                        f"to [{self.action_para['ex']}, {self.action_para['ey']}] on {self.app.name} APP")
            case ActionType.LIST_ALL_APP:
                return "list all Apps"
            case ActionType.FINISH:
                return "task finished"
            case ActionType.INVALID:
                return "invalid action"


class AndroidActionWrapper(ActionWrapper):

    def action(self, action: WrapperActType) -> ActType:
        """
        transform input `action dict` inferred by the agent to `Action` object
        :param action: action dict
        :return: Action object
        """
        action_obj = Action()
        action_type = ActionType.__getitem__(action["action"].upper())
        action_obj.action_type = action_type
        # app_level actions
        if action_type in [ActionType.START_APP, ActionType.STOP_APP]:
            pkg = self.env.phone.get_pkg_by_name(action["package"])
            if not pkg:
                raise AndroidActionException(
                    f"Cannot find APP {action['package']}. The APP name might be incorrect.")
            action_obj.app.name = pkg.name
            action_obj.app.package = pkg.package
        # component-leval actions
        elif action_type in [ActionType.CLICK, ActionType.LONG_CLICK, ActionType.DOUBLE_CLICK, ActionType.SET_TEXT]:
            if action["xpath"].startswith("//"):
                for node, xpath in self.env.cur_ui_xml_tree.node_to_xpath.items():
                    if action["xpath"] == xpath[0]:
                        action["xpath"] = xpath[1]
                        break
                action_obj.component.xpath = action["xpath"]
            else:
                if action["xpath"] not in self.env.cur_ui_xml_tree.node_to_xpath:
                    raise AndroidActionException(
                        f"Invalid node id {action['xpath']}. The node id might be incorrect.")
                else:
                    action_obj.component.xpath = self.env.cur_ui_xml_tree.node_to_xpath[action["xpath"]][1]
                action_obj.component.nearby_xpath = set(self.env.cur_ui_xml_tree.node_to_xpath[action["xpath"]][:2] + \
                                                        self.env.cur_ui_xml_tree.node_to_xpath[action["xpath"]][2])
                action_obj.component.name = self.env.cur_ui_xml_tree.node_to_name[action["xpath"]]
                app = self.env.phone.get_pkg_by_name(self.env.cur_ui_xml_tree.app_name)
                if not app:
                    app = APP(name=self.env.cur_ui_xml_tree.app_name)
                action_obj.app = app
            if action_type == ActionType.SET_TEXT:
                action_obj.action_para["text"] = action["text"]
        elif action_type == ActionType.FINISH:
            action_obj.action_para["text"] = action["text"]
        # do not need to parse args for none, finish, or system-level actions
        else:
            app = self.env.phone.get_pkg_by_name(self.env.cur_ui_xml_tree.app_name)
            if not app:
                app = APP(name=self.env.cur_ui_xml_tree.app_name)
            action_obj.app = app
        return action_obj

    def action_space(
            self,
    ) -> spaces.Space[ActType] | spaces.Space[WrapperActType]:
        space = spaces.Dict(
            {
                "action_type": spaces.Discrete(len(ActionType)),
                "action_para": spaces.Text(TEXT_MAX_LENGTH),
                "coords": spaces.Box(
                    np.array([0.0, 0.0], dtype=np.float32),
                    np.array([1.0, 1.0], dtype=np.float32),
                ),
                "app": spaces.Discrete(self.env.phone.num_apps),
                "activity": spaces.Discrete(self.env.phone.num_activities),
                "component": spaces.Discrete(self.env.phone.num_components),
            }
        )
        return space
