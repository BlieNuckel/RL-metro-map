from typing import TypeVar, Generic
from src.models.coordinates2d import Coordinates2d
from src.models.stop import Stop
from enum import Enum
import numpy as np
import cv2  # type: ignore

T = TypeVar("T")


class Grid(Generic[T]):
    """
    2D array container that provides (x,y) access instead of always having to access it by (y,x)

    Each field is made of a list too, and assignemnt automatically appends to the list, instead of overwriting
    """

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.grid: list[list[list[T | None]]] = [[[] for _ in range(width)] for _ in range(height)]

    def to_observation(self) -> list[int]:
        return [hash(tuple(x)) for row in self.grid for x in row]

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

    def render(self, color_map: dict[str, tuple[int, int, int]]) -> np.ndarray:

        # cv2.imshow("a", self.img)
        # cv2.waitKey(1)
        img = np.zeros((self.height * 2, self.width * 2, 3), dtype="uint8")

        for y, row in enumerate(self.grid):
            for x, field in enumerate(row):
                if len(field) == 0:
                    continue

                if isinstance(field[0], Stop):
                    cv2.rectangle(
                        img,
                        (x * 2, y * 2),
                        (x * 2 + 2, y * 2 + 2),
                        (255, 255, 255),
                        3,
                    )
                elif isinstance(field[0], str):
                    try:
                        cv2.rectangle(
                            img,
                            (x * 2, y * 2),
                            (x * 2 + 2, y * 2 + 2),
                            color_map[field[0]],
                            3,
                        )
                    except KeyError:
                        print(f"Field with value {field} has skipped rendering. No color value found.")
                else:
                    print(f"Non renderable grid item found: {field} at {(x, y)}. Implementation missing.")

        return img

    def __getitem__(self, key: tuple[int, int] | Coordinates2d) -> list[T | None]:
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

        self.grid[key.y][key.x].append(value)

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

    def __int__(self):
        return self.list().index(self)

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
