import os
import time

from stable_baselines3 import PPO
from src.environment import MetroMapEnv


def main() -> None:
    width = 700
    height = 300

    env = MetroMapEnv(width, height, "rgb_array")
    models_dir = f"./models/PPO-{int(time.time())}"
    log_dir = f"./logs/PPO-{int(time.time())}"

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    env.reset()

    model = PPO("MultiInputPolicy", env, verbose=0, tensorboard_log=log_dir, device="cuda")

    TIMESTEPS = 10000
    for i in range(1, 100):
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO", progress_bar=True)
        model.save(f"{models_dir}/{TIMESTEPS*i}")

    env.close()


if __name__ == "__main__":
    main()
