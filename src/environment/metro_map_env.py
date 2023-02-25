import gymnasium as gym
from gymnasium.core import RenderFrame
from typing import SupportsFloat, Any
from models import Stop, Grid
import numpy as np


class MetroMapEnv(gym.Env):
    __directions = [0, 1, 2, 3, 4, 5, 6, 7]
    __direction_map = {
        0: (0, -1),
        1: (1, -1),
        2: (1, 0),
        3: (1, 1),
        4: (0, 1),
        5: (-1, 1),
        6: (-1, 0),
        7: (-1, -1),
    }

    def __init__(
        self,
        width: int,
        height: int,
        lines: dict[str, list[Stop]],
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.action_space = gym.spaces.Discrete(6)
        self.render_mode = render_mode

        spaces: dict[str, gym.spaces.Space] = {
            "board": gym.spaces.Box(-np.inf, np.inf, (width, height)),
            "stops_remaining": gym.spaces.Box(0, np.inf, (1,)),
            "curr_direction": gym.spaces.Discrete(8),
        }
        self.observation_space = gym.spaces.Dict(spaces)

        self.grid = Grid[str](width, height)

    def step(self, action: gym.spaces.Discrete) -> tuple[gym.spaces.Dict, SupportsFloat, bool, bool, dict[str, Any]]:

        return super().step(action)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[gym.spaces.Dict, dict[str, Any]]:
        return super().reset(seed=seed, options=options)

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        if self.render_mode is None:
            return None

        return super().render()

    def close(self):
        return super().close()
