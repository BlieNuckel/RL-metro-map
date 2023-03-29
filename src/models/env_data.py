from dataclasses import dataclass
from src.models import Coordinates2d, Stop, Direction
from typing import Any


@dataclass(frozen=True)
class EnvData:
    lines: dict[str, list[Stop]]
    starting_stops: list[str]
    starting_positions: dict[str, tuple[Coordinates2d, Direction]]
    stop_angle_mapping: dict[str, dict[str, float]]
    line_color_map: dict[str, tuple[int, int, int]]
    turn_limits: tuple[int, int]
    stop_spacing: int


@dataclass(frozen=True)
class EnvDataDef:
    starting_stops: list[str]
    starting_positions: dict[str, tuple[Coordinates2d, Direction]]
    line_color_map: dict[str, tuple[int, int, int]]
    turn_limits: tuple[int, int]
    stop_spacing: int

    @staticmethod
    def from_json(json: dict[Any, Any]) -> "EnvDataDef":
        startin_stops = json["starting_stops"]

        starting_positions: dict[str, tuple[Coordinates2d, Direction]] = {}
        for key, value in json["starting_positions"].items():
            assert (
                isinstance(value[0], list) and len(value[0]) == 2
            ), "First value in starting_position must be a list of 2 integers (the starting coordinates)"

            assert isinstance(value[1], str), "Second value in starting_position must be a string"

            starting_positions[key] = (Coordinates2d(*value[0]), Direction.from_str_val(value[1]))

        line_color_map: dict[str, tuple[int, int, int]] = {}
        for key, value in json["line_color_map"].items():
            assert (
                isinstance(value, list) and len(value) == 3
            ), "Lines must be mapped to RGB values (3 values, between 0 and 255)"
            line_color_map[key] = tuple(value)  # type: ignore

        turn_limits = tuple(json["turn_limits"])
        stop_spacing = json["stop_spacing"]

        return EnvDataDef(startin_stops, starting_positions, line_color_map, turn_limits, stop_spacing)  # type: ignore # noqa: E501
