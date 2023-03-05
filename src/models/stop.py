from src.models.coordinates2d import Coordinates2d
from src.utils.math import angle_between_points


class Stop:
    def __init__(self, title: str, id: str, position: Coordinates2d) -> None:
        self.title = title
        self.id = id
        self.position = position
        self.__original_position = position

    def angle_to_stop(self, stop: "Stop") -> float:
        return angle_between_points(self.position, stop.position)

    def get_original_position(self) -> Coordinates2d:
        return self.__original_position

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, Stop):
            return False

        return self.id == __o.id

    def __hash__(self):
        return hash(self.id)
