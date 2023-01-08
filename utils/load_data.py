import os 
import numpy as np 

def load_data(filepath):
    """ Function loads csv files found in directory located at filepath. 

    Input:
    ++ filepath (str): filepath to location of directory you want to use

    Output:
    ++ list_of_files (list of str): list of names of all of the csv files loaded
    ++ all_arr (list of numpy array): the content of the csv files """

    list_of_files = os.listdir(filepath)
    all_arr = []

    # load csv files in directory, append to all_arr
    for file in list_of_files:
        if "csv" in file:
            arr = np.loadtxt(filepath+file, delimiter=",")
            all_arr.append(arr)

    return all_arr, list_of_files


