import numpy as np
from utils.extract_label import extract_label

def create_labels(dimensions, list_of_files):
    """ Function """

    all_labels = []

    for file in list_of_files:
        # create empty label array of correct dimensions
        labels = np.empty(dimensions[0])
        # print(labels.shape)

        # extract relevant label information
        part_num = extract_label(file)
        labels.fill(part_num)

        #label_array = labels.reshape(-1,1,1)
        label_array = labels.reshape(-1,1)
        all_labels.append(label_array)
        # print(label_array.shape)

    return all_labels

