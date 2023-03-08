from dataclasses import dataclass
from collections import deque
from src.models import Coordinates2d, Stop, Direction


@dataclass(frozen=True)
class EnvData:
    width: int
    height: int
    lines: dict[str, deque[Stop]]
    starting_positions: dict[str, tuple[Coordinates2d, Direction]]
    stop_angle_mapping: dict[str, dict[str, float]]
    line_color_map: dict[str, tuple[int, int, int]]
    turn_limits: tuple[int, int]
    stop_spacing: int
