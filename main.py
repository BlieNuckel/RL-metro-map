import os
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from src.environment import MetroMapEnv


def main() -> None:
    version = 3
    timesteps = 1000000
    width = 700
    height = 300
    models_dir = f"./generated_models/RewardFunctions_v{version}.zip"
    log_dir = f"./logs/RewardFunctions_v{version}_logs"

    os.makedirs(log_dir, exist_ok=True)

    env = MetroMapEnv(width, height, "rgb_array")
    check_env(env)

    monitor = Monitor(env)  # type: ignore
    monitor.reset()

    # eval_callback = EvalCallback(monitor, best_model_save_path=models_dir, eval_freq=100, n_eval_episodes=1)

    model = DQN("MultiInputPolicy", monitor, tensorboard_log=log_dir, device="cuda")

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
