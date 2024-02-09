import re
import subprocess
import time

import adbutils
import uiautomator2 as u2

from android_env2.actions import Action, ActionType
from android_env2.config import Settings


class Simulator:
    def __init__(
            self,
            config: Settings,
            **kwargs):
        self.emulator_path = config.emulator_path
        self.avd_name = config.avd_name
        self.adb_ip = config.adb_ip if config.adb_ip else "127.0.0.1"
        self.adb_port = config.adb_port
        self.emulator_name = config.emulator_name
        self.driver = None

        self.excluded_app = kwargs.get("excluded_app", [])

        self._prepare_device()

    def _prepare_device(self):
        devices = adbutils.adb.list()
        if len(devices) == 0:
            return
        if not self.driver:
            self.driver = u2.connect(self.emulator_name)
            print(f"uiautomator is connected to {self.emulator_name}...")

    def stop_avd(self):
        pass

    def reset(self):
        # self.driver.healthcheck()
        exclude_apps = ['com.github.uiautomator', 'com.github.uiautomator.test',
                        'com.google.android.apps.nexuslauncher', 'com.google.android.providers.media.module',
                        'com.android.remoteprovisioner', 'com.google.android.ext.services',
                        'com.google.android.permissioncontroller', 'com.android.bluetooth',
                        'com.google.android.apps.wellbeing', 'com.android.emulator.multidisplay',
                        'com.google.android.ims', 'com.google.android.adservices.api', 'com.android.vending',
                        'com.android.systemui', 'com.android.se']
        self.driver.app_stop_all(excludes=exclude_apps)
        self.driver.press("home")
        self.driver.set_fastinput_ime(True)

    def execute_action(self, action: Action):

        if action.action_type == ActionType.NONE or action.action_type == ActionType.FINISH:
            return

        elif action.action_type == ActionType.INSTALL_APP:
            self.driver.app_install(action.app.download_url)

        elif action.action_type == ActionType.START_APP:
            retry_time = 0
            while self.driver.app_current()["package"] != action.app.package and retry_time < 3:
                self.driver.app_start(action.app.package, use_monkey=True, wait=True)
                retry_time = retry_time + 1
                time.sleep(3)

        elif action.action_type == ActionType.STOP_APP:
            self.driver.app_stop(action.app.package)

        elif action.action_type == ActionType.CLICK:
            self.driver.xpath(action.component.xpath).click()

        elif action.action_type == ActionType.LONG_CLICK:
            self.driver.xpath(action.component.xpath).long_click()

        elif action.action_type == ActionType.DOUBLE_CLICK:
            self.driver.xpath(action.component.xpath).double_click()

        elif action.action_type == ActionType.SET_TEXT:
            self.driver.xpath(action.component.xpath).set_text(action.action_para["text"])

        elif action.action_type == ActionType.PRESS_BACK:
            self.driver.press("back")

        elif action.action_type == ActionType.PRESS_HOME:
            self.driver.press("home")

        elif action.action_type == ActionType.PRESS_ENTER:
            self.driver.press("enter")

        elif action.action_type == ActionType.SCREEN_ON:
            self.driver.screen_on()

        elif action.action_type == ActionType.SCREEN_OFF:
            self.driver.screen_off()

        elif action.action_type == ActionType.VOLUME_UP:
            self.driver.press("volume_up")

        elif action.action_type == ActionType.VOLUME_DOWN:
            self.driver.press("volume_down")

        elif action.action_type == ActionType.VOLUME_MUTE:
            self.driver.press("volume_mute")

        elif action.action_type == ActionType.SET_ORIENTATION:
            self.driver.set_orientation(action.action_para["orientation"])

        elif action.action_type == ActionType.FREEZE_ROTATION:
            self.driver.freeze_rotation()

        elif action.action_type == ActionType.UNFREEZE_ROTATION:
            self.driver.freeze_rotation(False)

        elif action.action_type == ActionType.SCREENSHOT:
            im = self.driver.screenshot()
            im.save(action.action_para["img_path"])

        elif action.action_type == ActionType.SWIPE_UP:
            self.driver.swipe_ext("up")

        elif action.action_type == ActionType.SWIPE_DOWN:
            self.driver.swipe_ext("down")

        elif action.action_type == ActionType.SWIPE_LEFT:
            self.driver.swipe_ext("left")

        elif action.action_type == ActionType.SWIPE_RIGHT:
            self.driver.swipe_ext("right")

        elif action.action_type == ActionType.SWIPE:
            self.driver.swipe(action.action_para['sx'],
                              action.action_para['sy'],
                              action.action_para['ex'],
                              action.action_para['ey'])

        elif action.action_type == ActionType.RECENT:
            self.driver.press("recent")

        elif action.action_type == ActionType.DRAG:
            self.driver.drag(action.action_para['sx'],
                             action.action_para['sy'],
                             action.action_para['ex'],
                             action.action_para['ey'])

        time.sleep(3)

    def current_app(self):
        return self.driver.app_current()

    def dump_ui_xml(self):
        xml = self.driver.dump_hierarchy()
        return xml

    def screenshot(self):
        im = self.driver.screenshot()
        return im

    def adb_shell(self, shell_cmd):
        output, exit_code = self.driver.shell(shell_cmd)

    def avd_log(self, log_path):
        pass
