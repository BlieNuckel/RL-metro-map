import gymnasium as gym
from gymnasium.core import RenderFrame
from typing import SupportsFloat, Any, cast
from src.models import Stop, Coordinates2d
from src.exceptions import OutOfBoundsException
from src.models.grid import Direction, Grid
from src.models.turn_queue import TurnQueue
from src.models.stop_adjacency import StopAdjacency
from src.environment import score_funcs, random_options
from src.utils.list import flat_map
from collections import deque
import numpy as np
import math
import cv2  # type: ignore


class MetroMapEnv(gym.Env):
    def __init__(
        self,
        width: int,
        height: int,
        # lines: dict[str, deque[Stop]],
        # starting_positions: dict[str, tuple[Coordinates2d, Direction]],
        # stop_angle_mapping: dict[str, dict[str, float]],
        # turn_limits: tuple[int, int],
        # stop_spacing: int,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.action_space = gym.spaces.Discrete(6)
        spaces: dict[str, gym.spaces.Space] = {
            "board": gym.spaces.Box(-np.inf, np.inf, (width * height,), dtype=np.int16),
            "stops_remaining_curr": gym.spaces.Box(0, np.inf, (1,), dtype=np.int16),
            "lines_remaining_all": gym.spaces.Box(0, np.inf, (1,), dtype=np.int16),
            "stops_remaining_all": gym.spaces.Box(0, np.inf, (1,), dtype=np.int16),
            "num_of_consecutive_overlaps": gym.spaces.Box(0, np.inf, (1,), dtype=np.int16),
            "num_of_turns": gym.spaces.Box(0, np.inf, (1,), dtype=np.int16),
            "max_turns": gym.spaces.Box(0, np.inf, (1,), dtype=np.int16),
            "curr_direction": gym.spaces.Discrete(8),
        }
        self.observation_space = gym.spaces.Dict(spaces)
        self.render_mode = render_mode

        self.grid_width = width
        self.grid_height = height

    @property
    def stops_remaining_curr(self) -> int:
        return len(self.lines[self.curr_line])

    @property
    def curr_line(self) -> str:
        return list(self.lines.keys())[self.curr_line_index]

    @property
    def stops_remaining_all(self) -> int:
        return len(flat_map(list(self.lines.values())))

    def step(self, action: int) -> tuple[dict[str, Any], SupportsFloat, bool, bool, dict[str, Any]]:
        terminated: bool = False
        truncated: bool = False
        reward: SupportsFloat = 0
        info: dict[str, Any] = {}

        match action:
            case 0:
                terminated, truncated, reward, info = self.__move_forward()
            case 1:
                terminated, truncated, reward, info = self.__turn_and_move(self.curr_direction.get_45_left())
            case 2:
                terminated, truncated, reward, info = self.__turn_and_move(self.curr_direction.get_90_left())
            case 3:
                terminated, truncated, reward, info = self.__turn_and_move(self.curr_direction.get_45_right())
            case 4:
                terminated, truncated, reward, info = self.__turn_and_move(self.curr_direction.get_90_right())
            case 5:
                terminated, truncated, reward, info = self.__place_stop()
            case _:
                pass

        if self.render_mode == "human":
            img = self.grid.render(self.line_color_map)
            cv2.imshow("a", img)
            cv2.waitKey(1)

        return self.__compile_observations(), reward, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed, options=options)

        info: dict[str, Any] = {}

        if options is None:
            env_data = random_options.generate_random_env_data(self.grid_width, self.grid_height)

            self.grid = Grid[str | Stop](env_data.width, env_data.height)
            self.lines = env_data.lines
            self.stop_spacing = env_data.stop_spacing
            self.real_stop_angles = env_data.stop_angle_mapping
            self.max_turns, self.steps_to_count_turns = env_data.turn_limits
            self.starting_positions = env_data.starting_positions
            self.line_color_map = env_data.line_color_map
        else:
            assert all(
                item in options.keys()
                for item in [
                    "lines",
                    "stop_spacing",
                    "stop_angle_mapping",
                    "turn_limits",
                    "starting_positions",
                    "line_color_map",
                    "grid_width",
                    "grid_height",
                ]
            ), "You must pass an options dict containing: 'lines', 'stop_spacing', 'stop_angle_mapping', 'turn_limits', 'line_color_map', 'grid_width', 'grid_height' and 'starting_positions'"  # noqa: E501

            self.grid = Grid[str | Stop](cast(int, options["grid_width"]), cast(int, options["grid_height"]))
            self.lines = cast(dict[str, deque[Stop]], options["lines"])
            self.stop_spacing = cast(int, options["stop_spacing"])
            self.real_stop_angles = cast(dict[str, dict[str, float]], options["stop_angle_mapping"])
            self.max_turns, self.steps_to_count_turns = cast(tuple[int, int], options["turn_limits"])
            self.starting_positions = cast(dict[str, tuple[Coordinates2d, Direction]], options["starting_positions"])
            self.line_color_map = cast(dict[str, tuple[int, int, int]], options["line_color_map"])

        self.steps_since_stop = 0
        self.consecutive_overlaps = 0
        self.stop_adjacency_map: StopAdjacency = StopAdjacency()
        self.recent_turns = TurnQueue(self.steps_to_count_turns)

        self.curr_line_index = 0
        self.lines_remaining_all = len(self.lines.keys()) - 1
        self.curr_position, self.curr_direction = self.starting_positions[self.curr_line]
        self.placed_stops: list[Stop] = []

        return (self.__compile_observations(), info)

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        if self.render_mode is None:
            return None

        if self.render_mode == "human":
            return None

        if self.render_mode == "rgb_array":
            return self.grid.render(self.line_color_map)  # type: ignore

        return super().render()

    def __compile_observations(self) -> dict[str, Any]:
        observations: dict[str, Any] = {}

        observations["board"] = np.array(self.grid.to_observation(), dtype=np.int16)
        observations["stops_remaining_curr"] = np.array([self.stops_remaining_curr], dtype=np.int16)
        observations["lines_remaining_all"] = np.array([self.lines_remaining_all], dtype=np.int16)
        observations["stops_remaining_all"] = np.array([self.stops_remaining_all], dtype=np.int16)
        observations["num_of_consecutive_overlaps"] = np.array([self.consecutive_overlaps], dtype=np.int16)
        observations["num_of_turns"] = np.array([self.recent_turns.num_of_turns()], dtype=np.int16)
        observations["max_turns"] = np.array([self.max_turns], dtype=np.int16)
        observations["curr_direction"] = int(self.curr_direction)

        return observations

    def __move_forward(self) -> tuple[bool, bool, float, dict[str, Any]]:
        terminated: bool = False
        truncated: bool = False
        reward: float = 0
        info: dict[str, Any] = {}

        self.curr_position += self.curr_direction.value

        if not self.grid.is_in_bounds(self.curr_position):
            reward += score_funcs.out_of_bounds()
            terminated = True

            return (terminated, truncated, reward, info)

        if not self.grid.is_empty(self.curr_position):
            self.consecutive_overlaps += 1
        else:
            self.consecutive_overlaps = 0

        self.grid[self.curr_position] = self.curr_line

        self.steps_since_stop += 1
        self.recent_turns.non_turn()

        reward += score_funcs.line_overlap(self.consecutive_overlaps)

        curr_line_start_point = self.starting_positions[self.curr_line][0]
        reward += score_funcs.promote_spreading(self.curr_position.distance_to(curr_line_start_point))

        return (terminated, truncated, reward, info)

    def __turn_and_move(self, new_direction: Direction) -> tuple[bool, bool, float, dict[str, Any]]:
        terminated: bool = False
        truncated: bool = False
        reward: float = 0
        info: dict[str, Any] = {}

        self.curr_direction = new_direction
        self.curr_position += self.curr_direction.value

        if not self.grid.is_in_bounds(self.curr_position):
            reward += score_funcs.out_of_bounds()
            terminated = True

            return (terminated, truncated, reward, info)

        if not self.grid.is_empty(self.curr_position):
            self.consecutive_overlaps += 1
        else:
            self.consecutive_overlaps = 0

        self.steps_since_stop += 1

        self.grid[self.curr_position] = self.curr_line

        self.recent_turns.turn()

        reward += score_funcs.line_overlap(self.consecutive_overlaps)
        reward += score_funcs.minimize_turns(self.recent_turns.num_of_turns())

        return (terminated, truncated, reward, info)

    def __place_stop(self) -> tuple[bool, bool, float, dict[str, Any]]:
        terminated: bool = False
        truncated: bool = False
        reward: float = 0
        info: dict[str, Any] = {}

        step_terminated, step_truncated = False, False
        step_reward: float = 0
        step_info: dict[str, Any] = {}

        # Move one step forward
        self.curr_position += self.curr_direction.value

        # Make sure we haven't stepped out of bounds
        if not self.grid.is_in_bounds(self.curr_position):
            reward += score_funcs.out_of_bounds()
            terminated = True

            # If we're out of bounds return an immediate, high punishment and terminate
            return (terminated, truncated, reward, info)

        try:
            if not self.grid.is_empty(self.curr_position):
                reward += score_funcs.stop_overlap()

            stop_to_place = self.lines[self.curr_line].popleft()
            self.grid[self.curr_position] = stop_to_place
            stop_to_place.position = self.curr_position

            is_stop_first = self.stop_adjacency_map.is_first(stop_to_place.id)

            if not is_stop_first:
                is_stop_placed_adjacent = self.stop_adjacency_map.is_adjacent(stop_to_place.id, self.curr_position)
                if is_stop_placed_adjacent:
                    self.__update_adjacency_map(stop_to_place)

                reward += score_funcs.stop_adjacency(is_stop_placed_adjacent)
            else:
                reward += score_funcs.stop_adjacency(is_stop_first)
                self.__update_adjacency_map(stop_to_place)

            reward += score_funcs.stop_distribution(self.steps_since_stop == self.stop_spacing)
            self.steps_since_stop = 0

            reward += self.__score_relative_stop_positions(stop_to_place)
            self.placed_stops.append(stop_to_place)

            if not self.__end_of_curr_line():
                step_terminated, step_truncated, step_reward, step_info = self.__move_forward()
                info.update(step_info)
            else:
                self.__handle_end_of_curr_line()

            terminated = self.__end_of_all_lines()

            return (terminated or step_terminated, truncated or step_truncated, reward + step_reward, info)
        except OutOfBoundsException:
            terminated = True
            reward += score_funcs.out_of_bounds()

            return (terminated, truncated, reward, info)

    def __update_adjacency_map(self, stop_to_place: Stop):
        self.stop_adjacency_map.remove_adjacency_position(stop_to_place.id, self.curr_position)

        left_of_stop = self.curr_position + self.curr_direction.get_90_left().value
        right_of_stop = self.curr_position + self.curr_direction.get_90_right().value

        if not isinstance(self.grid[left_of_stop], Stop):
            self.stop_adjacency_map.add_adjacency_position(stop_to_place.id, left_of_stop)

        if not isinstance(self.grid[right_of_stop], Stop):
            self.stop_adjacency_map.add_adjacency_position(stop_to_place.id, right_of_stop)

    def __score_relative_stop_positions(self, stop: Stop) -> float:
        scores: list[float] = []
        for placed_stop in self.placed_stops:
            if placed_stop == stop:
                continue

            real_angle = self.real_stop_angles[stop.id][placed_stop.id]
            new_angle = stop.angle_to_stop(placed_stop)

            scores.append(score_funcs.stop_relative_position(real_angle - new_angle))

        return math.fsum(scores)

    def __handle_end_of_curr_line(self) -> None:
        if not self.__end_of_curr_line():
            return

        if self.__end_of_all_lines():
            return

        self.curr_line_index += 1
        self.lines_remaining_all -= 1

        self.curr_position, self.curr_direction = self.starting_positions[self.curr_line]

    def __end_of_curr_line(self) -> bool:
        return self.stops_remaining_curr == 0

    def __end_of_all_lines(self) -> bool:
        return self.lines_remaining_all == 0
