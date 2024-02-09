from functools import lru_cache

from pydantic import BaseSettings


class Settings(BaseSettings):
    android_image: str = ""
    emulator_path: str = "<YOUR_EMULATOR_PATH>, e.g., XXXX\\android_sdk\\emulator\\emulator.exe"
    avd_name: str = "<ANDROID_VIRTUAL_DEVICE_NAME>"
    adb_ip: str = "127.0.0.1"
    adb_port: int = 5555
    emulator_name: str = "emulator-5554"

    early_stop: bool = True
    max_step: int = 50

    logger_path: str = "android_env_log/"

    phone_config_path = "app_configs/phone.yaml"


@lru_cache
def get_settings():
    settings = Settings()
    return settings
