import os.path
import re
from typing import List

import yaml


class Component:
    def __init__(self):
        self.name = None
        self.xpath = None
        self.nearby_xpath = set()
        self.description = None


class Activity:
    def __init__(
            self,
            name: str = None,
            description: str = None):
        self.name = name
        self.description = description
        self.components = []

    def load_from_dict(self, act):
        """
        load activities and define Component
        :param act:
        :return:
        """
        return

    @property
    def num_components(self):
        return len(self.components)


class APP:
    def __init__(
            self,
            download_url: str = None,
            package: str = None,
            name: str = None,
            description: str = None,
            activities: List[Activity] = None):
        self.download_url = download_url
        self.package = package
        self.name = name
        self.description = description
        self.activities = activities

    def retrieve_app_desc(self):
        google_play_url = "https://play.google.com/store/apps/details?id={}&hl=en_US"
        try:
            import trafilatura
            web = trafilatura.fetch_url(google_play_url.format(self.package))
            desc = trafilatura.extract(web)
        except ImportError:
            desc = ""
        return desc

    def load_from_yaml(self, yaml_path):
        """
        load the app description, activities and components in YAML file
        :param yaml_path:
        :return:
        """
        app_config = yaml.safe_load(open(yaml_path, "r"))
        self.package = app_config["app"]["package"]
        self.name = app_config["app"]["name"]
        if "description" in app_config["app"]:
            self.description = app_config["app"]["description"]
        else:
            self.description = self.retrieve_app_desc()
        self.activities = []
        activities = app_config["app"]["activities"]
        if activities:
            for act in activities:
                self.activities.append(Activity().load_from_dict(act))

    def dump_to_yaml(self, yaml_path):
        pass

    def update_property(self, k, v):
        pass

    def update_activity(self):
        pass

    @property
    def num_activities(self):
        return len(self.activities)

    @property
    def num_components(self):
        return sum([act.num_components for act in self.activities])


class User:
    def __init__(
            self,
            name: str = None,
            description: str = None):
        self.name = name
        self.description = description
        self.personality = None
        self.preference = None

    def update_from_history(self):
        """
        using LM to update user's personality and preference from behavior history
        :return:
        """
        pass


class UserTrace:
    def __init__(self):
        pass


class Phone:
    def __init__(self):
        self.user = User()
        self.apps = {}
        self.device_support = []
        self.device_info = {}

    def set_device_info(self, info):
        """
        {'udid': 'EMULATOR32X1X14X0-02:15:b2:00:00:00-sdk_gphone64_x86_64',
        'version': '13',
        'serial': 'EMULATOR32X1X14X0',
        'brand': 'google',
        'model': 'sdk_gphone64_x86_64',
        'hwaddr': '02:15:b2:00:00:00',
        'sdk': 33,
        'agentVersion': '0.10.0',
        'display': {'width': 320, 'height': 640},
        'battery': {'acPowered': False, 'usbPowered': False, 'wirelessPowered': False, 'status': 4, 'health': 2, 'present': True, 'level': 100, 'scale': 100, 'voltage': 5000, 'temperature': 250, 'technology': 'Li-ion'},
        'memory': {'total': 2013524, 'around': '2 GB'},
        'arch': '',
        'owner': None,
        'presenceChangedAt': '0001-01-01T00:00:00Z',
        'usingBeganAt': '0001-01-01T00:00:00Z',
        'product': None,
        'provider': None}
        """
        self.device_info = info

    def add_app(self, app: APP):
        self.apps[app.name] = app

    def remove_app(self, app: APP):
        self.apps.pop(app.name)

    def load_from_yaml(self, yaml_path):
        phone_config = yaml.safe_load(open(yaml_path, "r"))
        self.user.name = phone_config["user"]["name"]
        self.user.description = phone_config["user"]["self_introduction"]
        for app_name, app_path in phone_config["apps"].items():
            app_obj = APP()
            app_obj.load_from_yaml(os.path.join(os.path.dirname(yaml_path), app_path))
            self.add_app(app_obj)

    @property
    def num_apps(self):
        return len(self.apps)

    @property
    def num_activities(self):
        return sum([app.num_activities for app in self.apps])

    @property
    def num_components(self):
        return sum([act.num_components for app in self.apps for act in app.activities])

    def get_pkg_by_name(self, name) -> APP | None:
        if name in self.apps.keys():
            return self.apps[name]
        for app in self.apps.values():
            if app.package == name:
                return app
        return
