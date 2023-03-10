import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
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

    env = MetroMapEnv(width, height, "human")
    check_env(env)

    monitor = Monitor(env)  # type: ignore
    monitor.reset()

    # eval_callback = EvalCallback(monitor, best_model_save_path=models_dir, eval_freq=100, n_eval_episodes=1)

    model = PPO("MultiInputPolicy", monitor, tensorboard_log=log_dir, device="cuda")

    model.learn(
        # callback=eval_callback,
        total_timesteps=timesteps,
        reset_num_timesteps=False,
        log_interval=2,
        tb_log_name=f"RewardFunctions_v{version}",
        progress_bar=True,
    )

    model.save(models_dir)

    monitor.close()


if __name__ == "__main__":
    main()
