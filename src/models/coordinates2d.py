from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class Coordinates2d:
    x: int
    y: int

    def to_tuple(self) -> tuple[int, int]:
        return (self.x, self.y)

    def __add__(self, value: Union[tuple[int, int], "Coordinates2d"]) -> "Coordinates2d":
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
