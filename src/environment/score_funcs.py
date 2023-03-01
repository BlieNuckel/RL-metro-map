import math


def line_overlap(consecutive_overlaps: int) -> float:
    return -(2**consecutive_overlaps) + 1


def stop_overlap() -> float:
    return -200


def out_of_bounds() -> float:
    return -500


def stop_adjacency(stop_placed_adjacent: bool) -> float:
    return 10 if stop_placed_adjacent else -50


def stop_distribution(distributed_correctly: bool) -> float:
    return 6 if distributed_correctly else -20


def stop_relative_position(angle_difference: float) -> float:
    if angle_difference < 22.5:
        return 1
    else:
        return math.e ** ((-abs(angle_difference) + 22.5) / 4)


def minimize_turns(num_of_recent_turns: int) -> float:
    return -(4 ** (num_of_recent_turns - 3)) + 4
