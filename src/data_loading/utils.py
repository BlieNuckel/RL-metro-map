from pandas import DataFrame  # type: ignore
from src.constants.data import STOP_TITLE_COLUMN, STOP_NUMBER_COLUMN, X_COLUMN, Y_COLUMN


# TODO Collapsing still fails on example BP4456, 33008, and 33012.
# TODO These do not share a name but visually gets represented as "connected"
def remove_duplicate_stops(stops_per_route: DataFrame, stops: DataFrame) -> DataFrame:
    """
    Collapses all the stops that contain the same name into a single stop.
    """
    stops_per_route[STOP_NUMBER_COLUMN] = stops_per_route.groupby(STOP_TITLE_COLUMN)[STOP_NUMBER_COLUMN].transform(
        "first"
    )

    stops_per_route[X_COLUMN] = stops_per_route.groupby(STOP_TITLE_COLUMN)[X_COLUMN].transform("first")

    stops_per_route[Y_COLUMN] = stops_per_route.groupby(STOP_TITLE_COLUMN)[Y_COLUMN].transform("first")

    return stops_per_route
