from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from src.environment import MetroMapEnv
from typing import SupportsFloat
import cv2  # type: ignore


def main() -> None:
    version = 3
    width = 700
    height = 300
    models_dir = f"./generated_models/RewardFunctions_v{version}.zip"

    env = MetroMapEnv(width, height, "rgb_array")
    monitor = Monitor(env)  # type: ignore
    monitor.reset()

    model = PPO.load(models_dir, monitor, "cuda")
    terminated = False
    truncated = False
    obs, info = monitor.reset()
    reward: SupportsFloat = 0
    while not terminated or truncated:
        action, _ = model.predict(obs, deterministic=True)
        obs, step_reward, terminated, truncated, info = monitor.step(action)
        reward += step_reward  # type: ignore

    print(reward)
    cv2.imshow("a", env.render())
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
