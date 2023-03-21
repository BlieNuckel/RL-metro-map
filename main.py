import os
from sb3_contrib import QRDQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from src.environment import MetroMapEnv
from src.data_handling.load import load_training_data
from stable_baselines3.common.callbacks import EvalCallback


def main() -> None:
    version = 13
    timesteps = 2000000
    models_dir = f"./generated_models/RewardFunctions_v{version}"
    log_dir = f"./logs/RewardFunctions_v{version}_logs"

    os.makedirs(log_dir, exist_ok=True)

    training_data = load_training_data("./src/data/train_data.json")

    eval_env = MetroMapEnv(training_data=training_data)
    eval_monitor = Monitor(eval_env)  # type: ignore

    env = MetroMapEnv(render_mode="rgb_array", training_data=training_data)
    check_env(env)

    monitor = Monitor(env)  # type: ignore
    monitor.reset()

    eval_callback = EvalCallback(eval_monitor, best_model_save_path=models_dir, eval_freq=50000, n_eval_episodes=5)

    model = QRDQN(
        "MultiInputPolicy",
        monitor,
        tensorboard_log=log_dir,
        buffer_size=25000,
        device="cuda",
    )

    model.learn(
        callback=eval_callback,
        total_timesteps=timesteps,
        log_interval=2,
        tb_log_name=f"RewardFunctions_v{version}",
        progress_bar=True,
    )

    model.save(os.path.join(models_dir, "final_model"))

    monitor.close()


if __name__ == "__main__":
    main()
