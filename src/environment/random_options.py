from src.models.env_data import EnvDataDef, EnvData
from src.data_handling.parser import StopsPerRouteParser, StopsParser
from src.data_handling.data_modifiers import (
    remove_duplicate_stops,
    dataframe_as_routes_and_stops,
    extract_stop_angle_mappings,
)
from src.constants.data import STOP_NUMBER_COLUMN
from pandas import DataFrame  # type: ignore
import numpy as np

# import matplotlib.pyplot as plt  # type: ignore
# from src.utils.list import flat_map


class RandomOptions:
    def __init__(self, data: dict[str, EnvDataDef]) -> None:
        self.data = data

        self.routes_parser = StopsPerRouteParser()
        self.routes_parser.load_data("./src/data/stops_per_route.dbf")

        self.stops_parser = StopsParser()
        self.stops_parser.load_data("./src/data/stops.dbf")

    def generate_env_data(self, rand_gen: np.random.Generator | None = None, data_name: str | None = None) -> EnvData:
        if data_name is None:
            assert rand_gen is not None, "np.random.Generator must be passed if no data_name is passed"
            data_name, env_data_def = rand_gen.choice(np.array(list(self.data.items())), 1)

            max_turns = rand_gen.integers(2, 10)
            lookback_range = rand_gen.integers(10, 20)
            stop_distribution = rand_gen.integers(2, 20)
        else:
            env_data_def = self.data[data_name]

            max_turns, lookback_range = env_data_def.turn_limits
            stop_distribution = env_data_def.stop_spacing

        routes_df = self._load_and_parse_data(env_data_def.lines)

        routes_dict = dataframe_as_routes_and_stops(routes_df)

        stop_angle_mapping = extract_stop_angle_mappings(routes_dict)

        # normalized_routes: dict[str, list[Stop]] = dict(
        #     [
        #         (key, normalize_stop_positions(value, (0, env_data_def.width), (0, env_data_def.height)))
        #         for key, value in routes_dict.items()
        #     ]
        # )

        # all_stops = flat_map(routes_dict_deque.values())
        # plt.subplot(211)
        # plt.scatter([stop.position.x for stop in all_stops], [stop.position.y for stop in all_stops])

        # plt.subplot(212)
        # plt.scatter(
        #     [stop.get_original_position().x for stop in all_stops],
        #     [stop.get_original_position().y for stop in all_stops],
        # )

        # plt.show()

        return EnvData(
            env_data_def.width,
            env_data_def.height,
            routes_dict,
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
