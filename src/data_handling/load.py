from pandas import DataFrame  # type: ignore
import shapefile  # type: ignore


def load_dbf_table(path: str) -> DataFrame:
    assert path.endswith(".dbf"), "Please load just the .dbf file of the shape"

    shape_data = shapefile.Reader(path)
    fields = [x[0] for x in shape_data.fields][1:]
    records = [y[:] for y in shape_data.records()]

    return DataFrame(columns=fields, data=records)
