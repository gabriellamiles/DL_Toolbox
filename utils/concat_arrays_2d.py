import numpy as np

def concat_arrays_2d(all_arr, dim1):
    """ Concatenate 2D arrays to form matrix of size [-1, dim1].
    
    Input:
    ++ all_arr (list of numpy arrays): list of arrays to be concatenated
    ++ dim1 (int): size of dim1 in final array of shape [-1, dim1] 

    Output:
    ++ initialise (numpy array): array that is comprised of all arrays from all_arr concatenated and has        shaoe [-1, dim1] """

    initialise = np.empty((1,dim1))
    for arr in all_arr:
        initialise = np.concatenate((initialise, arr))
    initialise = initialise[1:,:]

    return initialise
