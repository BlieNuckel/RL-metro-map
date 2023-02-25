from .load import load_dbf_table
from .filters import filter_by_routes_and_stops, filter_by_stops
from pandas import DataFrame  # type: ignore


class StopsPerRouteParser:
    def __init__(self, stop_ids: list[str], data: DataFrame | None = None) -> None:
        self.stop_ids = stop_ids
        self.data = data

    def load_data(self, file_path: str) -> None:
        self.data = load_dbf_table(file_path)

    def filter_data(self) -> DataFrame:
        assert (
            self.data is not None
        ), "You must first load data either through the constructor or by calling 'load_data' on the parser"

        return filter_by_routes_and_stops(self.data, self.stop_ids)


class StopsParser:
    def __init__(self, stop_ids: list[str], data: DataFrame | None = None) -> None:
        self.stop_ids = stop_ids
        self.data = data

    def load_data(self, file_path: str) -> None:
        self.data = load_dbf_table(file_path)

    def filter_data(self) -> DataFrame:
        assert (
            self.data is not None
        ), "You must first load data either through the constructor or by calling 'load_data' on the parser"

        return filter_by_stops(self.data, self.stop_ids)
