from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from src.environment import MetroMapEnv
from typing import SupportsFloat
import cv2  # type: ignore
import sys
import os


def main() -> None:

    args = sys.argv

    width = 700
    height = 300
    models_dir = f"./generated_models/{args[1]}.zip"

    env = MetroMapEnv(width, height)
    monitor = Monitor(env)  # type: ignore
    monitor.reset()

    model = DQN.load(models_dir, monitor)

    terminated = False
    truncated = False
    obs, info = monitor.reset()
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
