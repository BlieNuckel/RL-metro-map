import os
from sb3_contrib import QRDQN
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.envs import BitFlippingEnv
from stable_baselines3.her import GoalSelectionStrategy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from src.data_handling.load import load_training_data
from stable_baselines3.common.callbacks import EvalCallback

from src.environment.metro_map_goal_env import MetroMapGoalEnv


def main() -> None:
    version = 100
    timesteps = 2000000
    models_dir = f"./goal_based/generated_models/RewardFunctions_v{version}"
    log_dir = f"./goal_based/logs/RewardFunctions_v{version}_logs"

    os.makedirs(log_dir, exist_ok=True)

    training_data = load_training_data("./src/data/train_data.json")

    eval_env = MetroMapGoalEnv(training_data=training_data)
    eval_monitor = Monitor(eval_env)  # type: ignore

    env = MetroMapGoalEnv(render_mode="rgb_array", training_data=training_data)
    check_env(env)

    monitor = Monitor(env)  # type: ignore
    monitor.reset()

    eval_callback = EvalCallback(eval_monitor, best_model_save_path=models_dir, eval_freq=50000, n_eval_episodes=5)
    BitFlippingEnv()
    goal_selection_strategy = GoalSelectionStrategy.FUTURE
    model = QRDQN(
        "MultiInputPolicy",
        monitor,
        tensorboard_log=log_dir,
        buffer_size=25000,
        device="cuda",
        replay_buffer_class=HerReplayBuffer,  # type: ignore
        replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy=goal_selection_strategy),
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
