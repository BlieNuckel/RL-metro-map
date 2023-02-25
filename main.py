from src.data_loading.parser import StopsPerRouteParser, StopsParser
from src.data_loading.utils import remove_duplicate_stops
from src.constants.data import STOP_NUMBER_COLUMN

from pandas import DataFrame  # type: ignore


def main():
    routes_data = __load_and_parse_data()


def __load_and_parse_data() -> DataFrame:
    routes_parser = StopsPerRouteParser(["17425", "17426"])

    routes_parser.load_data("./src/data/stops_per_route.dbf")
    routes_data = routes_parser.filter_data()

    stops_parser = StopsParser(list(routes_data[STOP_NUMBER_COLUMN].drop_duplicates()))
    stops_parser.load_data("./src/data/stops.dbf")
    stops_data = stops_parser.filter_data()

    return remove_duplicate_stops(routes_data, stops_data)


if __name__ == "__main__":
    main()
