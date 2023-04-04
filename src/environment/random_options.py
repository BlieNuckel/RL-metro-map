from src.models.env_data import EnvDataDef, EnvData
from src.data_handling.parser import StopsPerRouteParser, StopsParser
from src.data_handling.data_modifiers import (
    normalize_stop_positions,
    remove_duplicate_stops,
    dataframe_as_routes_and_stops,
    extract_stop_angle_mappings,
)
from src.constants.data import STOP_NUMBER_COLUMN
from pandas import DataFrame  # type: ignore
import numpy as np
from src.models.stop import Stop


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
            rand_idx = rand_gen.integers(0, len(self.data))
            data_name, env_data_def = list(self.data.items())[rand_idx]

            max_turns = rand_gen.integers(2, 10, dtype=int)
            lookback_range = rand_gen.integers(4, 10, dtype=int)
            stop_distribution = rand_gen.integers(2, 10, dtype=int)
        else:
            env_data_def = self.data[data_name]

            max_turns, lookback_range = env_data_def.turn_limits
            stop_distribution = env_data_def.stop_spacing

        routes_df = self._load_and_parse_data(env_data_def.starting_stops)

        routes_dict = dataframe_as_routes_and_stops(routes_df)

        stop_angle_mapping = extract_stop_angle_mappings(routes_dict)

        normalized_routes_dict: dict[str, list[Stop]] = normalize_stop_positions(
            routes_dict, env_data_def.starting_stops
        )

        final_routes_dict: dict[str, list[Stop]] = {
            key: normalized_routes_dict[key] for key in env_data_def.starting_positions.keys()
        }

        # all_stops = flat_map(normalized_routes_dict.values())
        # plt.subplot(111)
        # plt.scatter(
        #     [stop.position.x for stop in all_stops],
        #     [stop.position.y for stop in all_stops],
        #     c="r",
        # )
        # plt.scatter(
        #     [stop.get_original_position().x for stop in all_stops],
        #     [stop.get_original_position().y for stop in all_stops],
        #     c="g",
        # )

        # plt.show()

        return EnvData(
            final_routes_dict,
            env_data_def.starting_stops,
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
