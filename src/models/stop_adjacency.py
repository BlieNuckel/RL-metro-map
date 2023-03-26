from src.models import Coordinates2d
import math


class StopAdjacency:
    def __init__(self) -> None:
        self.__adjacency_map: dict[str, list[Coordinates2d]] = {}

    def is_adjacent(self, stop_id: str, position: Coordinates2d) -> bool:
        assert stop_id in self.__adjacency_map.keys(), "Check is_first before calling is_adjacent"

        return position in self.__adjacency_map[stop_id]

    def is_first(self, stop_id: str) -> bool:
        return stop_id not in self.__adjacency_map.keys()

    def adjacent_to_other(self, stop_id: str, position: Coordinates2d) -> bool:
        for id, adjacencies in self.__adjacency_map.items():
            if id == stop_id:
                continue

            if position in adjacencies:
                return True

        return False

    def get_nearest_adjacent(self, stop_id: str, position: Coordinates2d) -> Coordinates2d | None:
        assert stop_id in self.__adjacency_map.keys(), "Check is_first before calling is_adjacent"

        min_distance = math.inf
        best_adjacency: Coordinates2d | None = None
        for adjacency in self.__adjacency_map[stop_id]:
            distance = position.distance_to(adjacency)
            if distance < min_distance:
                min_distance = distance
                best_adjacency = adjacency

        return best_adjacency

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
