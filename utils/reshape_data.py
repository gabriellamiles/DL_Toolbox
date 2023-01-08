import numpy as np
import os

def reshape_data(all_arr, timestep, features):
    """ Function reshapes 2D timeseries data of [timestep, features] to 3D timeseries data
    of [samples, timestep, features].

    Input:
    ++ all_arr (list of numpy arrays): 2D timeseries data
    ++ timestep (int): number of frames to include in each timestep
    ++ features (int): number of features in data - should be equal to all_arr[0].shape[1]

    Output:
    ++ arr_reshape (list of numpy arrays): 3D timeseries data"""

    arr_reshape = []
    # reshape array data to correct
    for arr in all_arr:
        arr = arr.reshape(int(len(arr)/timestep), timestep, features)
        arr_reshape.append(arr)
        # print(arr.shape)

    # print(arr_reshape[0].shape)
    return arr_reshape



