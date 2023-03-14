from src.models.env_data import EnvDataDef, EnvData
from src.data_handling.parser import StopsPerRouteParser, StopsParser
from src.data_handling.data_modifiers import (
    remove_duplicate_stops,
    dataframe_as_routes_and_stops,
    extract_stop_angle_mappings,
)
from src.constants.data import STOP_NUMBER_COLUMN
from src.models import Stop
from pandas import DataFrame  # type: ignore
from collections import deque
from random import Random


class RandomOptions:
    def __init__(self, data: dict[str, EnvDataDef]) -> None:
        self.data = data

        self.routes_parser = StopsPerRouteParser()
        self.routes_parser.load_data("./src/data/stops_per_route.dbf")

        self.stops_parser = StopsParser()
        self.stops_parser.load_data("./src/data/stops.dbf")

    def generate_env_data(self, seed: int | None = None, data_name: str | None = None) -> EnvData:
        if data_name is None:
            random = Random(seed)
            data_name, env_data_def = random.choice(list(self.data.items()))

            max_turns = random.randint(2, 10)
            lookback_range = random.randint(10, 20)
            stop_distribution = random.randint(2, 20)
        else:
            env_data_def = self.data[data_name]

            max_turns, lookback_range = env_data_def.turn_limits
            stop_distribution = env_data_def.stop_spacing

        routes_df = self._load_and_parse_data(env_data_def.lines)
        routes_dict = dataframe_as_routes_and_stops(routes_df)

        stop_angle_mapping = extract_stop_angle_mappings(routes_dict)

        routes_dict_deque: dict[str, deque[Stop]] = dict([(key, deque(value)) for key, value in routes_dict.items()])

        return EnvData(
            env_data_def.width,
            env_data_def.height,
            routes_dict_deque,
            env_data_def.starting_positions,
            stop_angle_mapping,
            env_data_def.line_color_map,
            (max_turns, lookback_range),
            stop_distribution,
        )

    def _load_and_parse_data(self, starting_stops: list[str]) -> DataFrame:
        routes_data = self.routes_parser.filter_data(starting_stops)
        stops_data = self.stops_parser.filter_data(list(routes_data[STOP_NUMBER_COLUMN].drop_duplicates()))

        return remove_duplicate_stops(routes_data, stops_data, starting_stops)
