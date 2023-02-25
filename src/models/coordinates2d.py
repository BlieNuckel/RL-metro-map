from dataclasses import dataclass


@dataclass(frozen=True)
class Coordinates2d:
    x: float
    y: float

    def to_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, Coordinates2d):
            return False

        return self.x == __o.x and self.y == __o.y

    def __hash__(self) -> int:
        return hash(self.x) ^ hash(self.y)
