import math


def line_overlap(consecutive_overlaps: int) -> float:
    return _clamp_min(-60, -(2**consecutive_overlaps) + 1)


def stop_overlap() -> float:
    return -300


def out_of_bounds() -> float:
    return -500


def stop_adjacency(stop_placed_adjacent: bool) -> float:
    return 20 if stop_placed_adjacent else -50


def stop_distribution(distributed_correctly: bool) -> float:
    return 20 if distributed_correctly else -50


def stop_relative_position(angle_difference: float) -> float:
    if abs(angle_difference) < 22.5:
        return 1
    else:
        return math.e ** ((-abs(angle_difference) + 22.5) / 4)


def minimize_turns(num_of_recent_turns: int) -> float:
    return _clamp_min(-40, -(4 ** (num_of_recent_turns - 3)) + 4)


def promote_spreading(distance_from_start: float) -> float:
    return _clamp_min(0, 5 * math.log10(distance_from_start) + 5)


def _clamp_min(min_val: float, value: float) -> float:
    return max(min_val, value)


def _clamp(min_val: float, max_val: float, value: float) -> float:
    return max(min_val, min(max_val, value))
