from pathlib import Path
from functools import partial
from lsy_drone_racing.constants import FIRMWARE_FREQ
from lsy_drone_racing.utils import load_config
from safe_control_gym.utils.registration import make
from lsy_drone_racing.wrapper import DroneRacingWrapper

if __name__ == "__main__":
    config_path = "/home/tim/code/rl_experiments/lsy_drone_racing_rl/config/getting_started.yaml"
    config_path = Path(config_path)

    config = load_config(config_path)
    config.quadrotor_config.gui = False
    CTRL_FREQ = config.quadrotor_config["ctrl_freq"]
    # Create environment
    assert config.use_firmware, "Firmware must be used for the competition."
    pyb_freq = config.quadrotor_config["pyb_freq"]
    assert pyb_freq % FIRMWARE_FREQ == 0, "pyb_freq must be a multiple of firmware freq"
    config.quadrotor_config["ctrl_freq"] = FIRMWARE_FREQ

    print("Create 1")
    env_factory_1 = partial(make, "quadrotor", **config.quadrotor_config)
    firmware_env_1 = make("firmware", env_factory_1, FIRMWARE_FREQ, CTRL_FREQ)
    wrapper_1 = DroneRacingWrapper(firmware_env_1, terminate_on_lap=True)

    print("Create 2")
    env_factory_2 = partial(make, "quadrotor", **config.quadrotor_config)
    firmware_env_2 = make("firmware", env_factory_2, FIRMWARE_FREQ, CTRL_FREQ)
    wrapper_2 = DroneRacingWrapper(firmware_env_2, terminate_on_lap=True)

    print(id(wrapper_1), id(wrapper_2))