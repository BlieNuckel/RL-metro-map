from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.models.coordinates2d import Coordinates2d


class OutOfBoundsException(Exception):
    def __init__(self, position: Coordinates2d, message: str | None = None) -> None:
        self.position = position
        self.message = f"Position {self.position} is out of bound in the grid."

        if message is not None:
            self.message = message
