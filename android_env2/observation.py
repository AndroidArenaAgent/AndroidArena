import numpy as np
from gymnasium import spaces
from gymnasium.core import ObservationWrapper

from android_env2.constant import UTTERANCE_MAX_LENGTH, ASCII_CHARSET, FREQ_UNICODE_CHARSET


class ImageObsWrapper(ObservationWrapper):
    def observation(self, observation):
        img = self.env.simulator.screenshot()
        return img

    def observation_space(self):
        display = self.env.phone.device_info["display"]
        image_space = spaces.Box(low=0, high=255, shape=(display["width"], display["height"], 3), dtype=np.uint8)
        return image_space


class TextObsWrapper(ObservationWrapper):
    def observation(self, observation):
        xml_str = self.simulator.dump_ui_xml()
        app_info = self.env.simulator.current_app()
        package = app_info["package"]
        if "com.google.android.apps.nexuslauncher" == package:
            app_info["app_name"] = "home"
        else:
            app = self.env.phone.get_pkg_by_name(package)
            if not app:
                app_info["app_name"] = package.split(".")[-1]
            else:
                app_info["app_name"] = app.name
        xml_json = self.env.cur_ui_xml_tree.process(xml_str, app_info, level=2, str_type="plain_text")
        return xml_json

    def observation_space(self):
        text_space = spaces.Text(
            min_length=0,
            max_length=UTTERANCE_MAX_LENGTH,
            charset=ASCII_CHARSET + FREQ_UNICODE_CHARSET,
        )
        return text_space


class MixObsWrapper(TextObsWrapper):
    def observation(self, observation):
        xml_str = super().observation(observation)
        return {"text": xml_str, "image": self.env.simulator.screenshot()}

    def observation_space(self):
        text_space = spaces.Text(
            min_length=0,
            max_length=UTTERANCE_MAX_LENGTH,
            charset=ASCII_CHARSET + FREQ_UNICODE_CHARSET,
        )
        return text_space
