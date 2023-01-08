def same_length(all_arr, length):
    """ Function trims all arrays to the same (specified) length.

    Input:
    ++ all_arr (list of numpy arrays): list of numpy arrays 
    ++ length (int): length that you want to trim each numpy array too

    Output:
    ++ arr_trimmed (list of numpy arrays): same as all_arr, except after
    trimming operation """


    arr_trimmed = []

    for arr in all_arr:
        if len(arr) < length:
            print("Need to shorten array lengths!" + str(len(arr)))
        arr = arr[:length]
        arr_trimmed.append(arr)

    return arr_trimmed


