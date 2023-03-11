import math

COEFFICIENT_MAP = {
    "line_overlap": 8,
    "stop_overlap": 10,
    "out_of_bounds": 10,
    "stop_adjacency": 5,
    "stop_distribution": 5,
    "stop_relative_position": 8,
    "minimize_turns": 3,
    "promote_spreading": 5,
}


def line_overlap(consecutive_overlaps: int) -> float:
    if consecutive_overlaps <= 0:
        return 1

    return _clamp_min(-1, -(2 ** (2 * consecutive_overlaps - 3)) + 1)


def stop_overlap() -> float:
    return -1


def out_of_bounds() -> float:
    return -1


def stop_adjacency(stop_placed_adjacent: bool) -> float:
    return 2 / 5 if stop_placed_adjacent else -1


def stop_distribution(distributed_correctly: bool) -> float:
    return 2 / 5 if distributed_correctly else -1


def stop_relative_position(angle_difference: float) -> float:
    if abs(angle_difference) < 22.5:
        return 1
    else:
        return math.e ** ((-abs(angle_difference) + 22.5) / 4)


def minimize_turns(num_of_recent_turns: int) -> float:
    if num_of_recent_turns <= 0:
        return 1

    return _clamp_min(-1, -(2 ** (num_of_recent_turns - 4)) + 1)


def promote_spreading(distance_from_start: float) -> float:
    if distance_from_start <= 0:
        return 0

    return _clamp_min(0, 0.2 * math.log10(distance_from_start))


def _clamp_min(min_val: float, value: float) -> float:
    return max(min_val, value)


def _clamp(min_val: float, max_val: float, value: float) -> float:
    return max(min_val, min(max_val, value))
