from pandas import DataFrame, concat  # type: ignore
from src.constants.data import (
    STOP_TITLE_COLUMN,
    STOP_NUMBER_COLUMN,
    X_COLUMN,
    Y_COLUMN,
    ROUTE_RUN_NUMBER,
    RUN_NUMBER_COLUMN,
    STOP_SEQUENCE_NUMBER_COLUMN,
)
from src.models import Stop, Coordinates2d
from src.utils.list import flat_map
from typing import NamedTuple


# TODO Collapsing still fails on example BP4456, 33008, and 33012.
# TODO These do not share a name but visually gets represented as "connected"
def remove_duplicate_stops(stops_per_route: DataFrame, stops: DataFrame, excluded_stops: list[str]) -> DataFrame:
    """
    Collapses all the stops that contain the same name into a single stop.
    """
    excluded_rows = stops_per_route[stops_per_route[STOP_NUMBER_COLUMN].isin(excluded_stops)]
    stops_per_route.drop(excluded_rows.index, inplace=True)

    stops_per_route[STOP_NUMBER_COLUMN] = stops_per_route.groupby(STOP_TITLE_COLUMN)[STOP_NUMBER_COLUMN].transform(
        "first"
    )

    stops_per_route[X_COLUMN] = stops_per_route.groupby(STOP_TITLE_COLUMN)[X_COLUMN].transform("first")

    stops_per_route[Y_COLUMN] = stops_per_route.groupby(STOP_TITLE_COLUMN)[Y_COLUMN].transform("first")

    stops_per_route = concat([stops_per_route, excluded_rows])

    stops_per_route.sort_values(RUN_NUMBER_COLUMN, inplace=True)
    stops_per_route.sort_values(STOP_SEQUENCE_NUMBER_COLUMN, inplace=True)

    return stops_per_route


def dataframe_as_routes_and_stops(data: DataFrame) -> dict[str, list[Stop]]:
    def __generate_stop(row: NamedTuple) -> Stop:
        title = getattr(row, STOP_TITLE_COLUMN)
        stop_id = getattr(row, STOP_NUMBER_COLUMN)
        position = Coordinates2d(int(getattr(row, X_COLUMN)), int(getattr(row, Y_COLUMN)))
        return Stop(title, stop_id, position)

    routes_with_stops: dict[str, list[Stop]] = {}
    for row in data.itertuples():
        stop = __generate_stop(row)
        route_id = getattr(row, ROUTE_RUN_NUMBER)

        if route_id not in routes_with_stops.keys():
            routes_with_stops[route_id] = []

        routes_with_stops[route_id].append(stop)

    return routes_with_stops


def extract_stop_angle_mappings(data: dict[str, list[Stop]]) -> dict[str, dict[str, float]]:
    all_stops: list[Stop] = flat_map(list(data.values()))
    stop_angle_mapping: dict[str, dict[str, float]] = {}

    for stop in all_stops:
        relative_stop_angles: dict[str, float] = {}
        for related_stop in all_stops:
            if stop == related_stop:
                continue

            relative_stop_angles[related_stop.id] = stop.position.angle_to(related_stop.position)

        stop_angle_mapping[stop.id] = relative_stop_angles

    return stop_angle_mapping


def normalize_stop_positions(stops: list[Stop], x_limits: tuple[int, int], y_limits: tuple[int, int]) -> list[Stop]:
    # center_point = center_geolocation([stop.position for stop in stops])

    max_x = max([abs(stop.position.x) for stop in stops])
    max_y = max([abs(stop.position.y) for stop in stops])
    min_x = min([abs(stop.position.x) for stop in stops])
    min_y = min([abs(stop.position.y) for stop in stops])

    x_min_limit = x_limits[0]
    # x_max_limit = x_limits[1]
    y_min_limit = y_limits[0]
    # y_max_limit = y_limits[1]

    for stop in stops:
        norm_x = stop.position.x
        norm_y = stop.position.y

        norm_x -= (max_x + min_x) / 2
        norm_y -= (max_y + min_y) / 2

        x_scale = max_x - min_x + x_min_limit
        y_scale = max_y - min_y + y_min_limit

        # norm_x = (x_max_limit - x_min_limit) * ((stop.position.x - min_x) / (max_x - min_x)) + x_min_limit
        # norm_y = (y_max_limit - y_min_limit) * ((stop.position.y - min_y) / (max_y - min_y)) + y_min_limit

        stop.position = Coordinates2d(norm_x / x_scale, norm_y / y_scale)

    return stops


# def generate_starting_positions(routes_dict: dict[str, list[Stop]]) -> dict[str, tuple[Coordinates2d, Direction]]:
#     starting_stops: dict[str, Stop] = {}
#     routes_starting_stops: dict[str, list[str]] = {}
#     for line_id, stops in routes_dict.items():
#         if stops[0].id not in routes_starting_stops.keys():
#             routes_starting_stops[stops[0].id] = []
#             starting_stops[stops[0].id] = stops[0]

#         routes_starting_stops[stops[0].id].append(line_id)

#     spread = __calculate_spread(len(routes_starting_stops.keys()))

#     minNorm = -spread
#     maxNorm = spread
#     y_minEntry = min([stop.position.y for stop in starting_stops.values()])
#     y_maxEntry = max([stop.position.y for stop in starting_stops.values()])
#     x_minEntry = min([stop.position.x for stop in starting_stops.values()])
#     x_maxEntry = max([stop.position.x for stop in starting_stops.values()])

#     normalized_stop_coordinates: dict[str, Coordinates2d] = dict(
#         [
#             (
#                 stop_id,
#                 __normalize_coordinates(
#                     stop.position,
#                     Coordinates2d(x_minEntry, y_minEntry),
#                     Coordinates2d(x_maxEntry, y_maxEntry),
#                     Coordinates2d(minNorm, minNorm),
#                     Coordinates2d(maxNorm, maxNorm),
#                 ),
#             )
#             for stop_id, stop in starting_stops.items()
#         ]
#     )

#     print(normalized_stop_coordinates)


# def __calculate_spread(num_of_stops: int) -> int:
#     return num_of_stops // 2


# def __normalize_coordinates(
#     position: Coordinates2d,
#     min_entry: Coordinates2d,
#     max_entry: Coordinates2d,
#     min_norm: Coordinates2d,
#     max_norm: Coordinates2d,
# ) -> Coordinates2d:
#     x = __normalize(position.x, min_entry.x, max_entry.x, min_norm.x, max_norm.x)
#     y = __normalize(position.y, min_entry.y, max_entry.y, min_norm.y, max_norm.y)
#     return Coordinates2d(x, y)


# def __normalize(value: int, min_entry: int, max_entry: int, min_norm: int, max_norm: int) -> int:
#     mx = (value - min_entry) // (max_entry - min_entry)
#     preshift_normalized = mx * (max_norm - min_norm)
#     shifted_normalized = preshift_normalized + min_norm

#     return shifted_normalized
