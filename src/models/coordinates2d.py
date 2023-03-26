from dataclasses import dataclass
from typing import Union
import math
from src.utils.math import angle_between_points


@dataclass(frozen=True)
class Coordinates2d:
    x: float
    y: float

    def to_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)

    def distance_to(self, position: "Coordinates2d") -> float:
        return math.sqrt((self.x - position.x) ** 2 + (self.y - position.y) ** 2)

    def angle_to(self, position: "Coordinates2d") -> float:
        return angle_between_points(self, position)

    def __add__(self, value: Union[tuple[float, float], "Coordinates2d"]) -> "Coordinates2d":
        assert isinstance(value, tuple) or isinstance(value, Coordinates2d), (
            f"{value.__class__.__name__} cannot be added to Coordinates2d."
            + "Use a tuple with 2 ints or another Coordinates2d."
        )

        if isinstance(value, tuple):
            value = Coordinates2d(*value)

        return Coordinates2d(self.x + value.x, self.y + value.y)

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, Coordinates2d):
            return False

        return self.x == __o.x and self.y == __o.y

    def __hash__(self) -> int:
        return hash(self.x) ^ hash(self.y)

    def __str__(self) -> str:
        return str(self.to_tuple())
