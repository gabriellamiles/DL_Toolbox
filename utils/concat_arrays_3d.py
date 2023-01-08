import numpy as np

def concat_arrays_3d(all_arr, dim1, dim2):
    """ Concatenate 3D timeseries data to create a matrix of size [-1, dim1, dim2].

    Input:
    ++ all_arr(list of numpy arrays): list of numpy arrays to be concatenated
    ++ dim1 (int): size of dim1 in final array of shape [-1, dim1, dim2]
    ++ dim2 (int): size of dim2 in final array of shape [-1, dim1, dim2]

    Output:
    ++ initialise (numpy array): array that is comprised of all arrays from all_arr concatenated
       and has shape [-1, dim1, dim2] """

    initialise = np.empty((1,dim1,dim2))
    for arr in all_arr:
        initialise = np.concatenate((initialise, arr))

    initialise = initialise[1:,:]

    return initialise
