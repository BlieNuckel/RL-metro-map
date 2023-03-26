from src.models import Coordinates2d
import math


def angle_between_points(point1: Coordinates2d, point2: Coordinates2d) -> float:

    angle = math.degrees(
        math.atan2(-point1.y + point1.y, point1.x - point1.x) - math.atan2(-point2.y + point1.y, point2.x - point1.x)
    )

    return -angle % 360


def negative_mod(a: float, b: float) -> float:
    res = a % b
    return res if not res else res - b if a < 0 else res
