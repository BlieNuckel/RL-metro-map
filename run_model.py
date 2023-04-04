from sb3_contrib import QRDQN
from stable_baselines3.common.monitor import Monitor
from src.environment import MetroMapEnv
from src.data_handling.load import load_training_data
from typing import SupportsFloat
import sys
import os
from distutils.dir_util import copy_tree
import matplotlib.pyplot as plt  # type: ignore


def main() -> None:

    args = sys.argv
    version = args[1]
    map_id = args[2]

    assert version, "You must enter a version number"
    assert map_id, "You must enter a map_id"

    models_list = ["final_model", "best_model"]

    models_dir = f"./generated_models/RewardFunctions_v{version}/"
    log_save_dir = f"D:/GoogleDrive/Uni/ThesisProject/version_logs/RewardFunctions_v{version}"
    os.makedirs(log_save_dir, exist_ok=True)

    training_data = load_training_data("./src/data/train_data.json")

    assert map_id in training_data.keys(), "You must enter a valid map id, defined in the train_data.json file"

    env = MetroMapEnv(training_data=training_data, render_mode="rgb_array")
    monitor = Monitor(env, reset_keywords=tuple(["options"]))  # type: ignore

    for model_name in models_list:
        model = QRDQN.load(f"{models_dir}/{model_name}.zip", monitor, device="cuda")

        terminated = False
        truncated = False
        obs, info = monitor.reset(options={"env_data_def": map_id})
        reward: SupportsFloat = 0
        steps = 0

        while not terminated and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, step_reward, terminated, truncated, info = monitor.step(action)
            reward += step_reward  # type: ignore
            steps += 1
            os.system("cls")
            print(f"Step #: {steps}")
            print(f"Reward: {reward}")

        plt.imsave(f"{log_save_dir}/RewardFunctions_v{version}_{model_name}.png", env.render())  # type: ignore
        plt.imsave(f"./generated_maps/RewardFunctions_v{version}_{model_name}.png", env.render())  # type: ignore

        copy_tree(f"./generated_models/RewardFunctions_v{version}", f"{log_save_dir}/models")
        copy_tree(f"./logs/RewardFunctions_v{version}_logs", f"{log_save_dir}/RewardFunctions_v{version}_logs")


if __name__ == "__main__":
    main()
