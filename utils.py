
import numpy as np


def convert_arrays_to_numpy_in_dict(dictionary):
    '''Go through a dictionary and convert all lists to np arrays'''

    for key, value in dictionary.items():
        if isinstance(value, dict):
            convert_arrays_to_numpy_in_dict(value)
        elif isinstance(value, list):
            dictionary[key] = np.array(value, dtype=np.float64)


def standardize_angle(x):
    '''Converts an angle from (-inf, inf) to (-pi, pi]'''

    if x >= 0:
        x_bounded = x % (2*np.pi)
        if x_bounded > np.pi:
            return x_bounded - 2 * np.pi
        return x_bounded
    elif x < 0:
        x_bounded = x % (-2*np.pi)
        if x_bounded < -np.pi:
            return x_bounded + 2 * np.pi
        return x_bounded


def map_range(val, min0, max0, min1, max1):
    '''Maps a range [min0, max0] to [min1, max1]'''

    normalized_val = (val - min0) / (max0 - min0)
    return min1 + (max1 - min1) * normalized_val


def smooth_array(a, alpha=0.9):
    '''Compute moving average with exponential smoothing'''

    smoothed = [a[0]]
    for i in range(1, len(a)):
        smoothed.append(smoothed[-1] * alpha + (1-alpha) * a[i])
    return smoothed

    

# for x in [-1000, 234, -23, 4235, -234234, -39, -180, 180]:
#     pitch = standardize_angle(x * (np.pi/180))
#     print(pitch * (180 / np.pi))
#     print(int(round(map_range(pitch, -np.pi, np.pi, 0, 60 - 1))))

