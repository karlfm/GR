import json
from pathlib import Path
import numpy as np

# d = {"a": np.array([np.pi, 2, 3]), "b": np.array([4, 5, 6]), "c": 7j, "d": {"e": np.array([8, 9])}}

def json_numpy_serializer(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, complex):
        return [obj.real, obj.imag]
    raise TypeError("Type not serializable")

# Path("data.json").write_text(json.dumps(d, default=json_numpy_serializer))
# data_loaded = json.loads(Path("data.json").read_text())
# print(data_loaded)

def save_data(data, filename):
    Path(filename).write_text(json.dumps(data, default=json_numpy_serializer))