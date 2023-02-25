from dataclasses import dataclass
from src.models.coordinates2d import Coordinates2d


@dataclass
class Stop:
    title: str
    id: str
    position: Coordinates2d
