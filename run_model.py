from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from src.environment import MetroMapEnv
from src.data_handling.load import load_training_data
from typing import SupportsFloat
import cv2  # type: ignore
import sys
import os


def main() -> None:

    args = sys.argv

    models_dir = f"./generated_models/{args[1]}.zip"

    training_data = load_training_data("./src/data/train_data.json")

    env = MetroMapEnv(training_data=training_data, render_mode="human")
    monitor = Monitor(env, reset_keywords=tuple(["options"]))  # type: ignore

    model = DQN.load(models_dir, monitor, device="cuda")

    terminated = False
    truncated = False
    obs, info = monitor.reset(options={"env_data_def": "2_lines_wide_format"})
    reward: SupportsFloat = 0
    steps = 0

    while not terminated or truncated:
        action, _ = model.predict(obs, deterministic=True)
        obs, step_reward, terminated, truncated, info = monitor.step(action)
        reward += step_reward  # type: ignore
        steps += 1
        os.system("cls")
        print(f"Step #: {steps}")
        print(f"Reward: {reward}")

    print(reward)
    cv2.imshow("a", env.render())
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
