from typing import TypeVar, Generic
from src.models import Coordinates2d
from enum import Enum

T = TypeVar("T")


class Grid(Generic[T]):
    """
    2D array container that provides (x,y) access instead of always having to access it by (y,x)

    Each field is made of a list too, and assignemnt automatically appends to the list, instead of overwriting
    """

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.grid: list[list[T | None]] = [[None for _ in range(width)] for _ in range(height)]

    def to_observation(self) -> list[list[int]]:
        return [[hash(x) for x in row] for row in self.grid]

    def is_empty(self, position: tuple[int, int] | Coordinates2d) -> bool:
        if isinstance(position, tuple):
            position = Coordinates2d(*position)

        if not self.is_in_bounds(position):
            raise IndexError("Position outside of grid bounds")

        return self[position] is None

    def is_in_bounds(self, position: tuple[int, int] | Coordinates2d) -> bool:
        if isinstance(position, tuple):
            position = Coordinates2d(*position)

        return position.x <= self.width - 1 and position.y <= self.height - 1

    def __getitem__(self, key: tuple[int, int] | Coordinates2d) -> T | None:
        assert (isinstance(key, tuple) and len(key) == 2) or isinstance(
            key, Coordinates2d
        ), "You must use 2 dimensional access with the grid, nothing else"

        if isinstance(key, tuple):
            key = Coordinates2d(*key)

        if not self.is_in_bounds(key):
            raise IndexError("Position outside of grid bounds")

        return self.grid[key.y][key.x]

    def __setitem__(self, key: tuple[int, int] | Coordinates2d, value: T):
        assert (isinstance(key, tuple) and len(key) == 2) or isinstance(
            key, Coordinates2d
        ), "You must use 2 dimensional access with the grid, nothing else"

        if isinstance(key, tuple):
            key = Coordinates2d(*key)

        if not self.is_in_bounds(key):
            raise IndexError("Position outside of grid bounds")

        self.grid[key.y][key.x] = value

    def __str__(self) -> str:
        return_str = ""

        for row in self.grid:
            return_str += f"{row}\n"

        return return_str


class Direction(Enum):
    N = (0, -1)
    NE = (1, -1)
    E = (1, 0)
    SE = (1, 1)
    S = (0, 1)
    SW = (-1, 1)
    W = (-1, 0)
    NW = (-1, -1)

    @classmethod
    def list(cls) -> list["Direction"]:
        return list(cls)

    @property
    def value(self) -> tuple[int, int]:
        return super().value

    def get_45_right(self) -> "Direction":
        return self.__get_angle_by_index(1)

    def get_45_left(self) -> "Direction":
        return self.__get_angle_by_index(-1)

    def get_90_right(self) -> "Direction":
        return self.__get_angle_by_index(2)

    def get_90_left(self) -> "Direction":
        return self.__get_angle_by_index(-2)

    def __get_angle_by_index(self, index_offset: int) -> "Direction":
        directions = Direction.list()
        i = directions.index(self)
        return directions[(i + index_offset) % len(directions)]
