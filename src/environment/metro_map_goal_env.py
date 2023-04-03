from collections import deque
from typing import Any, Optional, SupportsFloat
import cv2  # type: ignore
from gymnasium.core import RenderFrame
from src.environment import score_funcs
from src.environment.env_util import GoalEnv  # type: ignore
from gymnasium import spaces
from src.environment.overlap import any_overlap, stop_overlap
from src.environment.render import render_map
from src.models import Stop, Coordinates2d, Direction
import numpy as np
from src.environment.random_options import RandomOptions

from src.models.env_data import EnvDataDef
from src.models.stop_adjacency import StopAdjacency
from src.utils.list import flat_map


class MetroMapGoalEnv(GoalEnv):
    def __init__(
        self,
        training_data: dict[str, EnvDataDef],
        max_steps: int = 15000,
        max_stops: int = 250,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.max_steps = max_steps
        self.max_stops = max_stops
        self.random_options = RandomOptions(training_data)

        self.action_space = spaces.Discrete(6)
        observation_space = spaces.Box(-np.inf, np.inf, (32,), dtype=np.float32)

        obs_wrap: dict[str, spaces.Space] = {
            "observation": observation_space,
            "desired_goal": spaces.Box(-np.inf, np.inf, (2,)),
            "achieved_goal": spaces.Box(-np.inf, np.inf, (2,)),
        }
        self.observation_space = spaces.Dict(obs_wrap)
        self.render_mode = render_mode

    @property
    def stops_remaining_curr(self) -> int:
        return len(self.lines[self.curr_line]) - (self.curr_stop_index + 1)

    @property
    def curr_line(self) -> str:
        return list(self.lines.keys())[self.curr_line_index]

    @property
    def stops_remaining_all(self) -> int:
        all_stops: list[Stop] = []
        for i, (_, stops) in enumerate(self.lines.items()):
            if i <= self.curr_line_index:
                continue
            all_stops.extend(stops)

        return len(all_stops) + self.stops_remaining_curr

    @property
    def curr_stop(self) -> Stop:
        return self.lines[self.curr_line][self.curr_stop_index]

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed, options=options)

        info: dict[str, Any] = {}

        if options is not None and "env_data_def" in options.keys():
            assert isinstance(
                options["env_data_def"], str
            ), "env_data_def key must be mapped to a valid name of a set of map data in train_data.json"

            env_data = self.random_options.generate_env_data(self.np_random, data_name=options["env_data_def"])
        else:
            if self.random_options is None:
                raise ValueError(
                    "You must either pass options, or instantiate the environment with a path to training data."
                )

            env_data = self.random_options.generate_env_data(self.np_random)

        self.placed_lines: dict[Coordinates2d, str] = {}
        self.placed_stops: dict[Coordinates2d, Stop] = {}
        self.lines = env_data.lines
        self.stop_spacing = env_data.stop_spacing
        self.real_stop_angles = env_data.stop_angle_mapping
        self.max_turns, self.steps_to_count_turns = env_data.turn_limits
        self.starting_positions = env_data.starting_positions
        self.line_color_map = env_data.line_color_map
        self.total_num_stops = len(flat_map(self.lines.values()))

        self.steps_since_stop = 0
        self.consecutive_overlaps = 0
        self.stop_adjacency_map: StopAdjacency = StopAdjacency()
        self.stop_in_adjacent_fields: np.ndarray = np.zeros((8,))
        self.line_in_adjacent_fields: np.ndarray = np.zeros((8,))
        self.out_of_bounds_in_adjacent_fields: np.ndarray = np.zeros((8,))
        self.recent_turns = deque([0 for _ in range(self.steps_to_count_turns)], maxlen=self.steps_to_count_turns)

        self.curr_line_index = 0
        self.lines_remaining_all = len(self.lines.keys()) - 1
        self.curr_position, self.curr_direction = self.starting_positions[self.curr_line]
        self.curr_stop_index = 0
        self.curr_stop_init_distance: float = self.curr_position.distance_to(self.curr_stop.position)
        self.curr_stop_prev_distance: float = 0

        self.total_steps = 0

        return (self.__compile_goal_observations(), info)

    def step(self, action: int) -> tuple[dict[str, Any], SupportsFloat, bool, bool, dict[str, Any]]:
        terminated: bool = False
        truncated: bool = False
        info: dict[str, Any] = {}

        self.line_in_adjacent_fields = np.zeros((8,))
        self.stop_in_adjacent_fields = np.zeros((8,))
        self.out_of_bounds_in_adjacent_fields = np.zeros((8,))

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

        # self.total_steps += 1
        # reward += score_funcs.time_alive(self.total_steps, self.total_num_stops, self.stop_spacing)

        # if self.total_steps > self.max_steps:
        #     terminated = True
        #     reward += score_funcs.out_of_bounds()

        if self.render_mode == "human":
            img = render_map(self.placed_lines, self.placed_stops, self.line_color_map)
            cv2.imshow("a", img)
            cv2.waitKey(1)

        obs = self.__compile_goal_observations()
        reward = self.compute_reward(obs["achieved_goal"], obs["desired_goal"], None)

        return obs, reward, terminated, truncated, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        results: list[int] = []

        if achieved_goal.ndim >= 2:
            for i in range(0, len(achieved_goal) - 1):
                if Coordinates2d(*achieved_goal[i]) == Coordinates2d(*desired_goal[i]):
                    results.append(0)
                else:
                    results.append(-1)

            return np.array(results)

        if Coordinates2d(*achieved_goal) == Coordinates2d(*desired_goal):
            return 0

        return -1

    def compute_truncated(self, achieved_goal, desired_goal, info):
        return super().compute_truncated(achieved_goal, desired_goal, info)

    def compute_terminated(self, achieved_goal, desired_goal, info):
        return super().compute_terminated(achieved_goal, desired_goal, info)

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        if self.render_mode is None:
            return None

        if self.render_mode == "human":
            return None

        if self.render_mode == "rgb_array":
            return render_map(self.placed_lines, self.placed_stops, self.line_color_map)  # type: ignore

        return super().render()

    def __compile_goal_observations(self) -> dict[str, Any]:
        observations: dict[str, Any] = {}

        observations["observation"] = self.__compile_env_observations()
        observations["desired_goal"] = np.array(self.curr_stop.get_original_position().to_tuple())
        observations["achieved_goal"] = np.array(self.curr_stop.position.to_tuple())

        return observations

    def __compile_env_observations(self) -> np.ndarray:
        observations: dict[str, Any] = {}

        observations["stop_in_adjacent_fields"] = list(self.stop_in_adjacent_fields)
        observations["line_in_adjacent_fields"] = list(self.line_in_adjacent_fields)
        observations["num_of_consecutive_overlaps"] = [self.consecutive_overlaps]
        observations["curr_direction"] = [int(self.curr_direction)]
        observations["curr_position"] = self.curr_position.to_tuple()
        observations["should_place_stop"] = [1 if self.curr_position.distance_to(self.curr_stop.position) <= 25 else 0]
        observations["next_stop_distance"] = [
            0 if self.curr_stop_index == 0 else self.curr_position.distance_to(self.curr_stop.position)
        ]
        observations["adjacent_to_same_stop"] = [
            1
            if self.stop_adjacency_map.is_first(self.curr_stop.id)
            or self.stop_adjacency_map.is_adjacent(self.curr_stop.id, self.curr_position + self.curr_direction.value)
            else 0
        ]
        observations["nearest_adjacent_position"] = [self.__get_distance_to_nearest_adjacent()]
        observations["adjacent_to_other_stop"] = [
            1
            if self.stop_adjacency_map.adjacent_to_other(
                self.curr_stop.id, self.curr_position + self.curr_direction.value
            )
            else 0
        ]

        return np.array(flat_map(observations.values()))

    def __move_forward(self, after_stop: bool = False) -> tuple[bool, bool, float, dict[str, Any]]:
        terminated: bool = False
        truncated: bool = False
        reward: float = 0
        info: dict[str, Any] = {}

        self.curr_position += self.curr_direction.value
        self.curr_stop.position = self.curr_position

        self.steps_since_stop += 1
        self.recent_turns.append(0)

        if any_overlap(self.placed_lines, self.placed_stops, self.curr_position):
            self.consecutive_overlaps += 1

            if stop_overlap(self.placed_stops, self.curr_position):
                reward += score_funcs.stop_overlap()
                terminated = True

            if self.consecutive_overlaps > 1:
                reward += score_funcs.line_overlap(self.consecutive_overlaps)
                terminated = True

            if terminated:
                return terminated, truncated, reward, info

        else:
            self.consecutive_overlaps = 0

        reward += score_funcs.line_overlap(self.consecutive_overlaps)
        # reward += score_funcs.stop_distribution(self.steps_since_stop, self.stop_spacing)
        dist_to_real_stop = self.curr_position.distance_to(self.curr_stop.position)
        if not after_stop:
            reward += score_funcs.distance_to_real_stop(
                dist_to_real_stop, self.curr_stop_prev_distance, self.curr_stop_init_distance
            )
        self.curr_stop_prev_distance = dist_to_real_stop

        self.placed_lines[self.curr_position] = self.curr_line

        self.__update_line_and_stop_adjacent()

        return terminated, truncated, reward, info

    def __turn_and_move(self, new_direction: Direction) -> tuple[bool, bool, float, dict[str, Any]]:
        terminated: bool = False
        truncated: bool = False
        reward: float = 0
        info: dict[str, Any] = {}

        self.recent_turns.append(self.curr_direction.get_difference(new_direction))
        self.curr_direction = new_direction

        terminated, truncated, reward, info = self.__move_forward()

        # reward += score_funcs.minimize_turns(sum(self.recent_turns))

        return (terminated, truncated, reward, info)

    def __place_stop(self) -> tuple[bool, bool, float, dict[str, Any]]:
        terminated: bool = False
        truncated: bool = False
        reward: float = 0
        info: dict[str, Any] = {}

        step_terminated, step_truncated = False, False
        step_reward: float = 0
        step_info: dict[str, Any] = {}

        self.curr_position += self.curr_direction.value

        if any_overlap(self.placed_lines, self.placed_stops, self.curr_position):
            reward += score_funcs.stop_overlap()
            terminated = True

            return terminated, truncated, reward, info

        stop_to_place = self.lines[self.curr_line][self.curr_stop_index]
        self.placed_stops[self.curr_position] = stop_to_place

        if self.curr_stop_index == 0:
            reward += score_funcs.stop_placed(0)
        else:
            reward += score_funcs.stop_placed(stop_to_place.position.distance_to(self.curr_position))

        stop_to_place.position = self.curr_position

        is_stop_first = self.stop_adjacency_map.is_first(stop_to_place.id)
        is_stop_placed_adjacent_wrong = self.stop_adjacency_map.adjacent_to_other(stop_to_place.id, self.curr_position)

        if not is_stop_first:
            is_stop_placed_adjacent = self.stop_adjacency_map.is_adjacent(stop_to_place.id, self.curr_position)
            if is_stop_placed_adjacent:
                self.__update_adjacency_map(stop_to_place)

            reward += score_funcs.stop_adjacency(is_stop_placed_adjacent_wrong, is_stop_placed_adjacent)
        else:
            if self.curr_stop_index == 0:
                reward += score_funcs.stop_adjacency(is_stop_placed_adjacent_wrong, is_stop_first)
            self.__update_adjacency_map(stop_to_place)

        self.steps_since_stop = 0

        self.__update_line_and_stop_adjacent()

        if not self.__end_of_curr_line():
            self.curr_stop_index += 1
            self.curr_stop_init_distance = self.curr_position.distance_to(self.curr_stop.position)
            self.curr_stop_prev_distance = 0
            step_terminated, step_truncated, step_reward, step_info = self.__move_forward(True)
            info.update(step_info)
        else:
            self.__handle_end_of_curr_line()
            reward += score_funcs.finished()

        terminated = self.__end_of_all_lines()

        return (terminated or step_terminated, truncated or step_truncated, reward + step_reward, info)

    def __update_adjacency_map(self, stop_to_place: Stop):
        self.stop_adjacency_map.remove_adjacency_position(stop_to_place.id, self.curr_position)

        left_of_stop = self.curr_position + self.curr_direction.get_90_left().value
        right_of_stop = self.curr_position + self.curr_direction.get_90_right().value

        if not any_overlap(self.placed_lines, self.placed_stops, left_of_stop):
            self.stop_adjacency_map.add_adjacency_position(stop_to_place.id, left_of_stop)

        if not any_overlap(self.placed_lines, self.placed_stops, right_of_stop):
            self.stop_adjacency_map.add_adjacency_position(stop_to_place.id, right_of_stop)

    def __update_line_and_stop_adjacent(self) -> None:
        for i, dir in enumerate(Direction.list()):
            check_pos = self.curr_position + dir.value

            if not any_overlap(self.placed_lines, self.placed_stops, check_pos):
                continue

            if stop_overlap(self.placed_stops, check_pos):
                self.stop_in_adjacent_fields[i] = 1
                continue

            self.line_in_adjacent_fields[i] = 1

    def __get_distance_to_nearest_adjacent(self) -> float:
        if self.stop_adjacency_map.is_first(self.curr_stop.id):
            return 0

        nearest_adjacent = self.stop_adjacency_map.get_nearest_adjacent(self.curr_stop.id, self.curr_position)

        if nearest_adjacent is None:
            return 0

        return self.curr_position.distance_to(nearest_adjacent)

    def __handle_end_of_curr_line(self) -> None:
        if not self.__end_of_curr_line():
            return

        if self.__end_of_all_lines():
            return

        self.curr_line_index += 1
        self.lines_remaining_all -= 1

        self.curr_stop_index = 0
        self.curr_position, self.curr_direction = self.starting_positions[self.curr_line]
        self.curr_stop_init_distance = self.curr_position.distance_to(self.curr_stop.position)
        self.curr_stop_prev_distance = 0

    def __end_of_curr_line(self) -> bool:
        return self.stops_remaining_curr == 0

    def __end_of_all_lines(self) -> bool:
        return self.lines_remaining_all == 0 and self.stops_remaining_all == 0
