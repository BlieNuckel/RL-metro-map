import json5  # type: ignore
from pandas import DataFrame  # type: ignore
import shapefile  # type: ignore
from src.models.env_data import EnvDataDef


def load_dbf_table(path: str) -> DataFrame:
    assert path.endswith(".dbf"), "Please load just the .dbf file of the shape"

    shape_data = shapefile.Reader(path)
    fields = [x[0] for x in shape_data.fields][1:]
    records = [y[:] for y in shape_data.records()]

    return DataFrame(columns=fields, data=records)


def load_training_data(data_path: str) -> dict[str, EnvDataDef]:
    env_data_result: dict[str, EnvDataDef] = {}

    with open(data_path) as json_file:
        data_dict = json5.load(json_file)

        assert isinstance(data_dict, dict), "Data to be loaded must be a dictionary of training_name -> env_data"

        for key, item in data_dict.items():
            assert isinstance(key, str), "Keys of training data must be strings"
            assert isinstance(item, dict), "Values of training data must be objects (in form of EnvData)"
            env_data_result[key] = EnvDataDef.from_json(item)

    return env_data_result
