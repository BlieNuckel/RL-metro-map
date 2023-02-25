from pandas import DataFrame  # type: ignore
import pandas as pd
from ..constants.data import ROUTE_NUMBER_COLUMN, STOP_NUMBER_COLUMN, RUN_NUMBER_COLUMN, STOP_SEQUENCE_NUMBER_COLUMN


def filter_by_routes_and_stops(stops_per_route: DataFrame, stop_ids: list[str]) -> DataFrame:
    route_datas = []

    route_and_stop_ids = __get_routes_from_stops(stops_per_route, stop_ids)

    for route_id, stop_id in route_and_stop_ids:
        route_data = __filter_by_route_id(stops_per_route, route_id)

        __sort_on_column(route_data, RUN_NUMBER_COLUMN)
        __sort_on_column(route_data, STOP_SEQUENCE_NUMBER_COLUMN)

        run_num, seq_num = __get_run_and_seq_nums(route_data, stop_id)

        __filter_by_run_number(route_data, run_num)
        __filter_by_after_seq_num(route_data, seq_num)

        route_datas.append(route_data)

    return pd.concat(route_datas)


def filter_by_stops(stops: DataFrame, stop_ids: list[str]):
    return stops[stops[STOP_NUMBER_COLUMN].isin(stop_ids)]


def __get_routes_from_stops(stops_per_route: DataFrame, stop_ids: list[str]) -> list[tuple[str, str]]:
    filtered_data = stops_per_route[stops_per_route[STOP_NUMBER_COLUMN].isin(stop_ids)]

    route_and_stop_ids = []

    for row in filtered_data.itertuples():
        route_and_stop_ids.append((getattr(row, ROUTE_NUMBER_COLUMN), getattr(row, STOP_NUMBER_COLUMN)))

    return route_and_stop_ids


def __filter_by_route_id(stops_per_route: DataFrame, route_id: str) -> DataFrame:
    return stops_per_route.drop(stops_per_route[stops_per_route[ROUTE_NUMBER_COLUMN] != route_id].index)


def __sort_on_column(route_data: DataFrame, column_name: str) -> None:
    route_data.sort_values(column_name, inplace=True)


def __get_run_and_seq_nums(route_data: DataFrame, stop_id: str) -> tuple[str, int]:
    stop_row = route_data[route_data[STOP_NUMBER_COLUMN] == stop_id].iloc[0]
    return (stop_row[RUN_NUMBER_COLUMN], stop_row[STOP_SEQUENCE_NUMBER_COLUMN])


def __filter_by_run_number(route_data: DataFrame, run_num: str) -> None:
    route_data.drop(route_data[route_data[RUN_NUMBER_COLUMN] != run_num].index, inplace=True)


def __filter_by_after_seq_num(route_data: DataFrame, seq_num: int) -> None:
    route_data.drop(route_data[route_data[STOP_SEQUENCE_NUMBER_COLUMN] <= seq_num].index, inplace=True)
