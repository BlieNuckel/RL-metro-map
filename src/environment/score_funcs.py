C_LINE_OVERLAP = 100
C_STOP_OVERLAP = 100
C_OUT_OF_BOUNDS = 100
C_STOP_ADJACENCY = 12
C_STOP_RELATIVE_POS = 11
C_DIST_TO_REAL_STOP = 4
C_STOP_DISTRIBUTION = 10
C_MINIMIZE_TURNS = 2
C_FINISHED = 50
C_TIME_ALIVE = 1


def line_overlap(consecutive_overlaps: int) -> float:
    if consecutive_overlaps <= 0:
        return C_LINE_OVERLAP * 0

    if consecutive_overlaps <= 1:
        return C_LINE_OVERLAP * -0.5

    return C_LINE_OVERLAP * -1
    # return C_LINE_OVERLAP * _clamp_min(-1, -(2 ** (2 * consecutive_overlaps - 3)) + 1)


def stop_overlap() -> float:
    return C_STOP_OVERLAP * -1


def max_steps_reached() -> float:
    return C_OUT_OF_BOUNDS * -1


# def stop_adjacency(stop_placed_adjacent_wrong: bool, stop_placed_adjacent: bool) -> float:
#     if stop_placed_adjacent_wrong:
#         return C_STOP_ADJACENCY * -1

#     return C_STOP_ADJACENCY * (1 if stop_placed_adjacent else -0.5)


# def stop_distribution(steps_since_stop: int, stop_distribution: int) -> float:
#     if steps_since_stop <= stop_distribution:
#         return C_STOP_DISTRIBUTION * _clamp(-1, 1, 0.25 * steps_since_stop)
#     else:
#         return C_STOP_DISTRIBUTION * _clamp(-1, 1, -steps_since_stop + stop_distribution + 1)

# return C_STOP_DISTRIBUTION * (2 / 5 if distributed_correctly else -1)


def stop_placed(distance_to_real_stop: float) -> float:
    if abs(distance_to_real_stop) <= 25:
        return C_STOP_RELATIVE_POS * 1

    return C_STOP_RELATIVE_POS * -1


# def stop_relative_position(angle_difference: float) -> float:
#     if abs(angle_difference) < 22.5:
#         return C_STOP_RELATIVE_POS * 1
#     else:
#         return C_STOP_RELATIVE_POS * (((math.e / 2) ** (-abs(angle_difference) + 24.75)) - 1)


def distance_to_real_stop(distance: float, prev_distance: float, init_distance: float) -> float:
    if distance < prev_distance:
        return C_DIST_TO_REAL_STOP * 0.8

    return C_DIST_TO_REAL_STOP * -1

    # if abs(distance) == 25:
    #     return C_DIST_TO_REAL_STOP * 1
    # else:
    # return C_STOP_RELATIVE_POS * ((math.e / 2) ** (-abs(distance) + 25))
    # return C_DIST_TO_REAL_STOP * ((prev_distance - distance) / init_distance)
    # return C_DIST_TO_REAL_STOP * (1 / (distance - 25))


# def minimize_turns(recent_turns_degrees: int) -> float:
#     if recent_turns_degrees <= 45:
#         return C_MINIMIZE_TURNS * 0
#     elif recent_turns_degrees <= 90:
#         return C_MINIMIZE_TURNS * -0.1
#     elif recent_turns_degrees <= 180:
#         return C_MINIMIZE_TURNS * -0.5
#     else:
#         return C_MINIMIZE_TURNS * -1

# if recent_turns_degrees <= 0:
#     return C_MINIMIZE_TURNS * 1  # 1 -> 0

# return C_MINIMIZE_TURNS * _clamp_min(-1, -(2 ** (recent_turns_degrees)) + 1)


# def promote_spreading(distance_from_start: float) -> float:
#     if distance_from_start <= 0:
#         return C_PROMOTE_SPREADING * 0

#     return C_PROMOTE_SPREADING * _clamp_min(0, 0.2 * math.log10(distance_from_start))


# def time_alive(time_step: int, total_stop_count: int, stop_distribution: int) -> float:
#     if time_step <= total_stop_count * stop_distribution:
#         return C_TIME_ALIVE * _clamp_max(1, 1 * time_step)

#     return C_TIME_ALIVE * -1


def finished() -> float:
    return C_FINISHED * 1


def _clamp_min(min_val: float, value: float) -> float:
    return max(min_val, value)


def _clamp_max(max_val: float, value: float) -> float:
    return min(max_val, value)


def _clamp(min_val: float, max_val: float, value: float) -> float:
    return max(min_val, min(max_val, value))
