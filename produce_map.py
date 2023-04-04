import cv2  # type: ignore
from sb3_contrib import QRDQN
from stable_baselines3.common.monitor import Monitor
from src.environment import MetroMapEnv
from src.data_handling.load import load_training_data
from typing import SupportsFloat
import sys
import os


def main() -> None:

    args = sys.argv
    version = args[1]
    map_id = args[2]
    model_name = args[3]

    assert version, "You must enter a version number"
    assert map_id, "You must enter a map_id"

    models_list = ["final_model", "best_model"]
    assert model_name in models_list, "Model choice invalid, must be 'final_model' or 'best_model'"

    models_dir = f"./generated_models/RewardFunctions_v{version}/"

    training_data = load_training_data("./src/data/train_data.json")

    assert map_id in training_data.keys(), "You must enter a valid map id, defined in the train_data.json file"

    env = MetroMapEnv(training_data=training_data, render_mode="rgb_array")
    monitor = Monitor(env, reset_keywords=tuple(["options"]))  # type: ignore

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

    cv2.imshow(f"{map_id} | {model_name} | v{version}", cv2.cvtColor(env.render(), cv2.COLOR_BGR2RGB))  # type: ignore
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
