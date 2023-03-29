import numpy as np
from src.models import Coordinates2d, Stop
import cv2  # type: ignore


def render_map(
    lines: dict[Coordinates2d, str], stops: dict[Coordinates2d, Stop], color_map: dict[str, tuple[int, int, int]]
) -> np.ndarray:
    try:
        min_x = min([pos.x for pos in lines.keys()] + [pos.x for pos in stops.keys()])  # -1499
        min_y = min([pos.y for pos in lines.keys()] + [pos.y for pos in stops.keys()])  # -1324
        max_x = max([pos.x for pos in lines.keys()] + [pos.x for pos in stops.keys()])  # 1422
        max_y = max([pos.y for pos in lines.keys()] + [pos.y for pos in stops.keys()])  # 1530
    except ValueError:
        min_x, max_x, min_y, max_y = 0, 0, 0, 0

    margin = 10

    width = abs(max_x - min_x) + margin
    height = abs(max_y - min_y) + margin

    width = max(width, 1)
    height = max(height, 1)

    img = np.zeros((height * 2, width * 2, 3), dtype="uint8")  # type: ignore

    for pos, line_id in lines.items():
        pos_x = (pos.x + abs(min_x) + margin // 2) * 2
        pos_y = (pos.y + abs(min_y) + margin // 2) * 2
        cv2.rectangle(
            img,
            (pos_x, pos_y),
            (pos_x + 1, pos_y + 1),
            color_map[line_id],
            thickness=cv2.FILLED,
        )

    for pos, _ in stops.items():
        pos_x = (pos.x + abs(min_x) + margin // 2) * 2
        pos_y = (pos.y + abs(min_y) + margin // 2) * 2
        cv2.rectangle(
            img,
            (pos_x, pos_y),
            (pos_x + 1, pos_y + 1),
            (255, 255, 255),
            thickness=cv2.FILLED,
        )

    return img
