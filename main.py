import os
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from src.environment import MetroMapEnv


def main() -> None:
    version = 2
    timesteps = 1000000
    width = 700
    height = 300
    models_dir = f"./generated_models/RewardFunctions_v{version}"
    log_dir = f"./logs/RewardFunctions_v{version}"

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    env = MetroMapEnv(width, height, "rgb_array")
    env.reset()

    eval_callback = EvalCallback(env, best_model_save_path=models_dir)

    model = DQN("MultiInputPolicy", env, verbose=0, tensorboard_log=log_dir, device="cuda")
    model.learn(
        callback=eval_callback,
        total_timesteps=timesteps,
        reset_num_timesteps=False,
        tb_log_name=f"RewardFunctions_v{version}",
        progress_bar=True,
    )

    env.close()


if __name__ == "__main__":
    main()
