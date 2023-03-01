from src.models import Coordinates2d


class StopAdjacency:
    def __init__(self) -> None:
        self.__adjacency_map: dict[str, list[Coordinates2d]] = {}

    def is_adjacent(self, stop_id: str, position: Coordinates2d) -> bool:
        return position in self.__adjacency_map[stop_id]

    def remove_adjacency_position(self, stop_id: str, position: Coordinates2d) -> None:
        self.__adjacency_map[stop_id].remove(position)

    def add_adjacency_position(self, stop_id: str, position: Coordinates2d) -> None:
        self.__adjacency_map[stop_id].append(position)

    def __getitem__(self, key: str) -> list[Coordinates2d]:
        return self.__adjacency_map[key]
