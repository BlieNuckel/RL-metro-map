import os
from typing import SupportsFloat
from stable_baselines3.common.monitor import Monitor
from src.data_handling.load import load_training_data
from src.environment.metro_map_env import MetroMapEnv
import cv2  # type: ignore
import sys


def main() -> None:
    training_data = load_training_data("./src/data/train_data.json")

    env = MetroMapEnv(training_data=training_data, render_mode="rgb_array")
    monitor = Monitor(env, reset_keywords=tuple(["options"]))  # type: ignore

    terminated = False
    truncated = False
    obs, info = monitor.reset(options={"env_data_def": "2_lines_wide_format"})
    reward: SupportsFloat = 0
    steps = 0

    while not terminated or truncated:
        cv2.imshow("Metro Map Game", env.render())
        input_key = cv2.waitKey(0)

        action = handle_input(input_key)
        if action == -1:
            continue
        obs, step_reward, terminated, truncated, info = monitor.step(action)
        reward += step_reward  # type: ignore

        os.system("cls")
        print(f"Step #: {steps}")
        print(f"Step Reward: {step_reward}")
        print(f"Total Reward: {reward}")
        print("Observations ------------")
        for key, value in obs.items():
            print(f"{key}: {value}")
        steps += 1


def handle_input(input_key: int) -> int:
    if input_key == 27:
        cv2.destroyAllWindows()
        sys.exit()

    match input_key:
        case 119:
            return 0
        case 113:
            return 1
        case 97:
            return 2
        case 101:
            return 3
        case 100:
            return 4
        case 32:
            return 5
        case _:
            return -1


if __name__ == "__main__":
    main()
