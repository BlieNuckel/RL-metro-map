from src.models import Coordinates2d


class StopAdjacency:
    def __init__(self) -> None:
        self.__adjacency_map: dict[str, list[Coordinates2d]] = {}

    def is_adjacent(self, stop_id: str, position: Coordinates2d) -> bool:
        assert stop_id in self.__adjacency_map.keys(), "Check is_first before calling is_adjacent"

        return position in self.__adjacency_map[stop_id]

    def is_first(self, stop_id: str) -> bool:
        return stop_id not in self.__adjacency_map.keys()

    def remove_adjacency_position(self, stop_id: str, position: Coordinates2d) -> None:
        if stop_id not in self.__adjacency_map.keys():
            return

        self.__adjacency_map[stop_id].remove(position)

    def add_adjacency_position(self, stop_id: str, position: Coordinates2d) -> None:
        if stop_id not in self.__adjacency_map.keys():
            self.__adjacency_map[stop_id] = []

        self.__adjacency_map[stop_id].append(position)

    def __getitem__(self, key: str) -> list[Coordinates2d]:
        return self.__adjacency_map[key]
