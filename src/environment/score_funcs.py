import math

C_LINE_OVERLAP = 20
C_STOP_OVERLAP = 20
C_OUT_OF_BOUNDS = 20
C_STOP_ADJACENCY = 6
C_STOP_RELATIVE_POS = 4
C_STOP_DISTRIBUTION = 4.5
C_MINIMIZE_TURNS = 10
C_PROMOTE_SPREADING = 1.5
C_FINISHED = 20
C_TIME_ALIVE = 1


def line_overlap(consecutive_overlaps: int) -> float:
    if consecutive_overlaps <= 0:
        return C_LINE_OVERLAP * 1

    if consecutive_overlaps <= 1:
        return C_LINE_OVERLAP * 0.5

    return C_LINE_OVERLAP * -1
    # return C_LINE_OVERLAP * _clamp_min(-1, -(2 ** (2 * consecutive_overlaps - 3)) + 1)


def stop_overlap() -> float:
    return C_STOP_OVERLAP * -1


def out_of_bounds() -> float:
    return C_OUT_OF_BOUNDS * -1


def stop_adjacency(stop_placed_adjacent: bool) -> float:
    return C_STOP_ADJACENCY * (2 / 5 if stop_placed_adjacent else -1)


def stop_distribution(distributed_correctly: bool) -> float:
    return C_STOP_DISTRIBUTION * (2 / 5 if distributed_correctly else -1)


def stop_relative_position(angle_difference: float) -> float:
    if abs(angle_difference) < 22.5:
        return C_STOP_RELATIVE_POS * 1
    else:
        return C_STOP_RELATIVE_POS * (math.e ** ((-abs(angle_difference) + 22.5) / 4))


def minimize_turns(num_of_recent_turns: int, max_recent_turns: int) -> float:
    if num_of_recent_turns <= 0:
        return C_MINIMIZE_TURNS * 1

    return C_MINIMIZE_TURNS * _clamp_min(-1, -(2 ** (num_of_recent_turns - max_recent_turns)) + 1)


# def promote_spreading(distance_from_start: float) -> float:
#     if distance_from_start <= 0:
#         return C_PROMOTE_SPREADING * 0

#     return C_PROMOTE_SPREADING * _clamp_min(0, 0.2 * math.log10(distance_from_start))


def time_alive(time_step: int) -> float:
    if time_step <= 0:
        return C_TIME_ALIVE * 0

    return C_TIME_ALIVE * _clamp_max(1, 0.1 * time_step)


def finished() -> float:
    return C_FINISHED * 1


def _clamp_min(min_val: float, value: float) -> float:
    return max(min_val, value)


def _clamp_max(max_val: float, value: float) -> float:
    return min(max_val, value)


def _clamp(min_val: float, max_val: float, value: float) -> float:
    return max(min_val, min(max_val, value))
