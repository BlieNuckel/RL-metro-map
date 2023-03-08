from src.models.env_data import EnvData
from src.data_handling.parser import StopsPerRouteParser, StopsParser
from src.data_handling.data_modifiers import (
    remove_duplicate_stops,
    dataframe_as_routes_and_stops,
    extract_stop_angle_mappings,
)
from src.constants.data import STOP_NUMBER_COLUMN
from src.models import Stop, Coordinates2d, Direction
from pandas import DataFrame  # type: ignore
from collections import deque


# TODO Make this use actual random data entries (probably put together manually)
def generate_random_env_data(width: int, height: int, seed: int | None = None) -> EnvData:

    routes_df = _load_and_parse_data(["17425", "17426"])
    routes_dict = dataframe_as_routes_and_stops(routes_df)

    starting_positions = {
        "194_1": (Coordinates2d(width // 2 - 2, height // 2), Direction.W),
        "194_2": (Coordinates2d(width // 2 + 2, height // 2), Direction.E),
        "356_1": (Coordinates2d(width // 2 - 2, height // 2), Direction.W),
        "356_2": (Coordinates2d(width // 2 + 2, height // 2), Direction.E),
        "358_1": (Coordinates2d(width // 2 + 2, height // 2), Direction.E),
        "358_2": (Coordinates2d(width // 2 - 2, height // 2), Direction.W),
    }
    line_color_map: dict[str, tuple[int, int, int]] = {
        "194_1": (255, 0, 0),
        "194_2": (255, 0, 0),
        "356_1": (0, 255, 0),
        "356_2": (0, 255, 0),
        "358_1": (0, 0, 255),
        "358_2": (0, 0, 255),
    }

    stop_angle_mapping = extract_stop_angle_mappings(routes_dict)

    routes_dict_deque: dict[str, deque[Stop]] = dict([(key, deque(value)) for key, value in routes_dict.items()])

    return EnvData(width, height, routes_dict_deque, starting_positions, stop_angle_mapping, line_color_map, (4, 7), 4)


def _load_and_parse_data(starting_stops: list[str]) -> DataFrame:
    routes_parser = StopsPerRouteParser(starting_stops)

    routes_parser.load_data("./src/data/stops_per_route.dbf")
    routes_data = routes_parser.filter_data()

    stops_parser = StopsParser(list(routes_data[STOP_NUMBER_COLUMN].drop_duplicates()))
    stops_parser.load_data("./src/data/stops.dbf")
    stops_data = stops_parser.filter_data()

    return remove_duplicate_stops(routes_data, stops_data, starting_stops)
