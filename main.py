from src.data_handling.parser import StopsPerRouteParser, StopsParser
from src.data_handling.data_modifiers import (
    remove_duplicate_stops,
    dataframe_as_routes_and_stops,
    extract_stop_angle_mappings,
)
from src.constants.data import STOP_NUMBER_COLUMN
from src.environment import MetroMapEnv
from src.models import Stop, Coordinates2d, Direction
from pandas import DataFrame  # type: ignore
from collections import deque
from stable_baselines3.common.env_checker import check_env


def main() -> None:
    width = 700
    height = 300

    routes_df = __load_and_parse_data(["17425", "17426"])
    routes_dict = dataframe_as_routes_and_stops(routes_df)

    for key, value in routes_dict.items():
        print(f"{key}: {value[0].id}")

    starting_positions = {
        "194_1": (Coordinates2d(width // 2 - 2, height // 2), Direction.W),
        "194_2": (Coordinates2d(width // 2 + 2, height // 2), Direction.E),
        "356_1": (Coordinates2d(width // 2 - 2, height // 2), Direction.W),
        "356_2": (Coordinates2d(width // 2 + 2, height // 2), Direction.E),
        "358_1": (Coordinates2d(width // 2 + 2, height // 2), Direction.E),
        "358_2": (Coordinates2d(width // 2 - 2, height // 2), Direction.W),
    }
    stops_angle_mapping = extract_stop_angle_mappings(routes_dict)

    routes_dict_deque: dict[str, deque[Stop]] = dict([(key, deque(value)) for key, value in routes_dict.items()])

    env = MetroMapEnv(500, 500, routes_dict_deque, starting_positions, stops_angle_mapping, (4, 7), 8)

    check_env(env)


def __load_and_parse_data(starting_stops: list[str]) -> DataFrame:
    routes_parser = StopsPerRouteParser(starting_stops)

    routes_parser.load_data("./src/data/stops_per_route.dbf")
    routes_data = routes_parser.filter_data()

    stops_parser = StopsParser(list(routes_data[STOP_NUMBER_COLUMN].drop_duplicates()))
    stops_parser.load_data("./src/data/stops.dbf")
    stops_data = stops_parser.filter_data()

    return remove_duplicate_stops(routes_data, stops_data, starting_stops)


if __name__ == "__main__":
    main()
