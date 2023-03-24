import math
from src.models.coordinates2d import Coordinates2d
from typing import TypeVar, Generic
from src.exceptions import OutOfBoundsException
from src.models.stop import Stop
from enum import Enum
import numpy as np
import cv2  # type: ignore
import hashlib

T = TypeVar("T")


class Grid(Generic[T]):
    """
    2D array container that provides (x,y) access instead of always having to access it by (y,x)

    Each field is made of a list, and assignemnt automatically appends to the list, instead of overwriting
    """

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.grid: list[list[list[T]]] = [[[] for _ in range(width)] for _ in range(height)]

    def to_observation(self) -> list[int]:
        observation_list = []

        for row in self.grid:
            for x in row:
                if len(x) <= 0:
                    observation_list.append(0)
                    continue

                string_to_hash = x[0]
                if isinstance(x[0], Stop):
                    string_to_hash = x[0].id  # type: ignore

                hashed_string = int(hashlib.sha1(string_to_hash.encode("utf-8")).hexdigest(), 16) % (  # type:ignore
                    10**8
                )
                observation_list.append(hashed_string)

        return observation_list

    def is_empty(self, position: tuple[int, int] | Coordinates2d) -> bool:
        if isinstance(position, tuple):
            position = Coordinates2d(*position)

        if not self.is_in_bounds(position):
            raise OutOfBoundsException(position)

        return len(self[position]) == 0

    def is_in_bounds(self, position: tuple[int, int] | Coordinates2d) -> bool:
        if isinstance(position, tuple):
            position = Coordinates2d(*position)

        return 0 <= position.x < self.width and 0 <= position.y < self.height

    def expand_grid(self, direction: "Direction") -> None:
        pass

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
                        (x * 2 + 1, y * 2 + 1),
                        (255, 255, 255),
                        thickness=cv2.FILLED,
                    )
                elif isinstance(field[0], str):
                    try:
                        cv2.rectangle(
                            img,
                            (x * 2, y * 2),
                            (x * 2 + 1, y * 2 + 1),
                            color_map[field[0]],
                            thickness=cv2.FILLED,
                        )
                    except KeyError:
                        print(f"Field with value {field} has skipped rendering. No color value found.")
                else:
                    print(f"Non renderable grid item found: {field} at {(x, y)}. Implementation missing.")

        return img

    def __direction_of_outside_bounds(self, position: Coordinates2d) -> "Direction":
        out_of_bounds_directions: list[Direction] = []

        if position.x >= self.width:
            out_of_bounds_directions.append(Direction.E)
        elif position.x < 0:
            out_of_bounds_directions.append(Direction.W)

        if position.y >= self.height:
            out_of_bounds_directions.append(Direction.S)
        elif position.y < 0:
            out_of_bounds_directions.append(Direction.N)

        if len(out_of_bounds_directions) > 1:
            return out_of_bounds_directions[0].combine(out_of_bounds_directions[1])

        return out_of_bounds_directions[0]

    def __getitem__(self, key: tuple[int, int] | Coordinates2d) -> list[T]:
        assert (isinstance(key, tuple) and len(key) == 2) or isinstance(
            key, Coordinates2d
        ), "You must use 2 dimensional access with the grid, nothing else"

        if isinstance(key, tuple):
            key = Coordinates2d(*key)

        if not self.is_in_bounds(key):
            raise OutOfBoundsException(key)

        try:
            return self.grid[math.floor(key.y)][math.floor(key.x)]
        except IndexError:
            print("Index error occurred in grid, with following data:")
            print(f"Position: {key}")
            print(f"Out of bounds reported: {self.is_in_bounds(key)}")
            print(f"Size is: {self.width}x{self.height}")
            raise OutOfBoundsException(key)

    def __setitem__(self, key: tuple[int, int] | Coordinates2d, value: T):
        assert (isinstance(key, tuple) and len(key) == 2) or isinstance(
            key, Coordinates2d
        ), "You must use 2 dimensional access with the grid, nothing else"

        if isinstance(key, tuple):
            key = Coordinates2d(*key)

        if not self.is_in_bounds(key):
            raise OutOfBoundsException(key)

        try:
            self.grid[math.floor(key.y)][math.floor(key.x)].append(value)
        except IndexError:
            print("Index error occurred in grid, with following data:")
            print(f"Position: {key}")
            print(f"Out of bounds reported: {self.is_in_bounds(key)}")
            print(f"Size is: {self.width}x{self.height}")
            raise OutOfBoundsException(key)

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

    @classmethod
    def from_str_val(cls, str_val: str) -> "Direction":
        match str_val:
            case "N":
                return Direction.N
            case "NE":
                return Direction.NE
            case "E":
                return Direction.E
            case "SE":
                return Direction.SE
            case "S":
                return Direction.S
            case "SW":
                return Direction.SW
            case "W":
                return Direction.W
            case "NW":
                return Direction.NW
            case _:
                raise ValueError("The string value must be N, NE, E, SE, S, SW, W, or NW")

    @staticmethod
    def direction_is_not_combined(direction: "Direction") -> bool:
        return Direction.list().index(direction) % 2 == 0

    @staticmethod
    def from_degree(degrees: float) -> "Direction":
        """Returns the nearest direction based on ``degrees`` starting from 0 in the unit circle."""
        direction_angles = {
            0: Direction.E,
            45: Direction.NE,
            90: Direction.N,
            135: Direction.NW,
            180: Direction.W,
            225: Direction.SW,
            270: Direction.S,
            315: Direction.SE,
        }
        for direction_angle in direction_angles.keys():
            if direction_angle - 22.5 <= degrees <= direction_angle + 22.5:
                return direction_angles[direction_angle]

        raise ValueError(f"The degree value passed in ({degrees}) could not be parsed into a direction.")

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

    def get_difference(self, direction: "Direction") -> int:
        directions = Direction.list()
        i = directions.index(self)
        j = directions.index(direction)

        return abs(i - j) * 45

    def combine(self, direction: "Direction") -> "Direction":
        assert Direction.direction_is_not_combined(direction), "Cannot combine already combined direction."
        assert self.__is_combineable(direction), "Cannot combine the two directions."

        if (self == Direction.N and direction == Direction.E) or (self == Direction.E and direction == Direction.N):
            return Direction.NE
        if (self == Direction.N and direction == Direction.W) or (self == Direction.W and direction == Direction.N):
            return Direction.NW
        if (self == Direction.S and direction == Direction.E) or (self == Direction.E and direction == Direction.S):
            return Direction.SE
        if (self == Direction.S and direction == Direction.W) or (self == Direction.W and direction == Direction.S):
            return Direction.SW

        raise ValueError()

    def __is_combineable(self, direction: "Direction") -> bool:
        return (
            (self == Direction.N or self == Direction.S) and (direction == Direction.W or direction == Direction.E)
        ) or ((self == Direction.E or self == Direction.W) and (direction == Direction.N or direction == Direction.S))

    def __get_angle_by_index(self, index_offset: int) -> "Direction":
        directions = Direction.list()
        i = directions.index(self)
        return directions[(i + index_offset) % len(directions)]
