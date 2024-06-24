
import numpy as np


def convert_arrays_to_numpy_in_dict(dictionary):
    for key, value in dictionary.items():
        if isinstance(value, dict):
            convert_arrays_to_numpy_in_dict(value)
        elif isinstance(value, list):
            dictionary[key] = np.array(value, dtype=np.float64)
