from src.models import Coordinates2d
import math


def angle_between_points(point1: Coordinates2d, point2: Coordinates2d) -> float:
    angle = math.degrees(math.atan2(point1.y - point2.y, point1.x - point2.x))

    return angle % 360
