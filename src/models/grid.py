from typing import TypeVar, Generic

T = TypeVar("T")


class Grid(Generic[T]):
    def __init__(self, width: int, height: int) -> None:
        self.grid: list[list[list[T]]] = [[[] for _ in range(width)] for _ in range(height)]

    def get_field(self, x: int, y: int) -> list[T]:
        return self.grid[y][x]

    def draw_in_field(self, x: int, y: int, value: T):
        self.grid[y][x].append(value)
