
from lsy_drone_racing.environment import make_env, start_from_scratch


if __name__ == "__main__":
    config = "tests/test_config.yaml"

    config = start_from_scratch(config)
    env = make_env(config, rank=0)()

    env.increase_environment_complexity()