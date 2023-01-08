# Gabriella Miles, Farscope PhD Student, Bristol Robotics Laboratory
# Date created: 6th August 2020

from utils.load_data import load_data
from utils.same_length import same_length
from utils.concat_arrays_3d import concat_arrays_3d
from utils.concat_arrays_2d import concat_arrays_2d
from utils.create_labels import create_labels
from utils.reshape_data import reshape_data
from utils.extract_label import extract_label
import models
import numpy as np
import sys
import os
import tensorflow 
import keras
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from copy import deepcopy

def create_classifier(name, input_shape, nb_classes, output_directory,variable):
    """ Function calls a deep learning classifier of type specified by name.

    Input:
    ++ name (str): name of model to be created, see README for correct inputs
    ++ input shape (tuple): serves to specify input of first layer of keras model, should be 
                            in format (trainX.shape[1], trainX.shape[2])
    ++ nb_classes (int): number of classes for classification problem
    ++ output_directory (str): output_directory !!! write this out properly.

    Outputs:
    ++ keras model of type found in file specified by name """

    if name=='lstm':
        from models import lstm
        return lstm.Classifier(input_shape, nb_classes, output_directory, variable, verbose=True)
    if name=='fcn':
        from models import fcn
        return fcn.Classifier(input_shape, nb_classes, output_directory, variable, verbose=True) 
    if name=='blstm':
        from models import b_lstm
        return b_lstm.Classifier(input_shape, nb_classes, output_directory, variable, verbose=True)
    if name=='cnnlstm':
        from models import cnn_lstm
        return cnn_lstm.Classifier(input_shape, nb_classes, output_directory, variable, verbose=True)
    if name=='mlp':
        from models import mlp
        return mlp.Classifier(input_shape, nb_classes, output_directory, variable, verbose=True)
    if name=='rlstm':
        from models import r_lstm
        return r_lstm.Classifier(input_shape, nb_classes, output_directory, variable, verbose=True)
    if name=='tlenet':
        from models import tlenet
        return tlenet.Classifier(input_shape, nb_classes, output_directory, variable, verbose=True)

def create_ml_classifier(name):
    """ Create a machine learning model of type specified by name. 

    Inputs:
    ++ name (str): name of model to be created, see README for correct inputs

    Outputs:
    ++ sklearn model of type specified by name"""

    if name=='lr':
        return LogisticRegression(verbose=1, n_jobs=-1)
    if name=='knn':
        return KNeighborsClassifier(n_jobs=-1)
    if name=='dt':
        return DecisionTreeClassifier()
    if name=='svm':
        return SVC()
    if name=='rf':
        return RandomForestClassifier()
    if name=='gb':
        return GradientBoostingClassifier()

def test_train_trad(classifier, trainX, trainy, testX, testy):
    """ Train sklearn model on data, and return results with accuracy ascertained via cross-fold 
    validation. 

    Inputs:
    ++ classifier (sklearn model): machine learning model to be trained
    ++ trainX (numpy array): data to train on 
    ++ trainy (numpy array): labels to train on
    ++ testX (numpy array): data to test on
    ++ testy (numpy array): labels to test on"""

    scores = cross_val_score(classifier, trainX, trainy, scoring='accuracy', cv=4, n_jobs=-1)
    m, std = np.mean(scores)*100, np.std(scores)*100
    print("Results:", str(m), str(std))

    return m, std

def split_dataset(all_arr, list_of_files):

    train_arr = []
    test_arr = []
    count = 0
    trials_in_train_set = []
    counter = 0
    training_files = []
    testing_files = []

    for name in list_of_files:
        arr = all_arr[counter]
        part_num = int(name[-10:-7]) # participant number

        if trials_in_train_set.count(part_num) >= 3:
            test_arr.append(arr)
            testing_files.append(name)
        else:
            train_arr.append(arr)
            trials_in_train_set.append(part_num)
            training_files.append(name)

        counter += 1

    return train_arr, test_arr, training_files, testing_files 

def crossval_split(all_arr, list_of_files):
    fold_0 = []
    fold_1 = []
    fold_2 = []
    fold_3 = []

    name_0 = []
    name_1 = []
    name_2 = []
    name_3 = []

    count = 0

    for name in list_of_files:
        arr = all_arr[count]
        if len(name) >= 44:
            fold = int(name[-12:-10])
        else:
            fold = int(name[-6:-4]) 
        if fold == 0:
            fold_0.append(arr)
            name_0.append(name)
        elif fold == 1:
            fold_1.append(arr)
            name_1.append(name)
        elif fold == 2:
            fold_2.append(arr)
            name_2.append(name)
        elif fold == 3:
            fold_3.append(arr)
            name_3.append(name)
        else:
            print("Unrecognised value detected. Review crossval_split.")
        count += 1

    print("Fold data:")
    print(len(fold_0), len(fold_1), len(fold_2), len(fold_3))
    print(len(name_0), len(name_1), len(name_2), len(name_3))
    print(fold_0[0].shape)

    labels_1 = create_labels(fold_0[0].shape, name_0) 
    labels_2 = create_labels(fold_0[0].shape, name_1)
    labels_3 = create_labels(fold_0[0].shape, name_2)
    labels_4 = create_labels(fold_0[0].shape, name_3)

    print("Label data:")
    print(len(labels_1), len(labels_2), len(labels_3), len(labels_4))
    print(labels_1[0].shape)

    # combine the data in a single fold into a concatenated array
    concatenated_folds = []
    for fold in [fold_0, fold_1, fold_2, fold_3]:
        arr = concat_arrays_3d(fold, 100, 4) # timestep, features 
        concatenated_folds.append(arr)

    print(concatenated_folds[0].shape)

    # combine the label data of a single fold into a concatenated array
    concatenated_labels = []
    for labels in [labels_1, labels_2, labels_3, labels_4]:
        lab = concat_arrays_2d(labels, 1)
        concatenated_labels.append(lab)

    print(concatenated_labels[0].shape)

    # arrange training and test sets for each fol
    cv1 = cv_train_test(concatenated_folds, concatenated_labels,0)
    cv2 = cv_train_test(concatenated_folds, concatenated_labels,1)
    cv3 = cv_train_test(concatenated_folds, concatenated_labels,2)
    cv4 = cv_train_test(concatenated_folds, concatenated_labels,3)

    return cv1, cv2, cv3, cv4

def cv_train_test(list_of_folds, list_of_labels, fold_num):

    folds = deepcopy(list_of_folds)
    labels = deepcopy(list_of_labels)

    testX = folds.pop(fold_num)
    testy = labels.pop(fold_num) 
    trainX = concat_arrays_3d(folds, 100, 4) # timestep, features
    trainy = concat_arrays_2d(labels, 1)

    print(trainX.shape, trainy.shape, testX.shape, testy.shape)

    return [trainX, trainy, testX, testy]

if __name__=='__main__':

    # extract data location from sys.argv[1]
    data_dir = str(sys.argv[1])
    all_arr, list_of_files = load_data(data_dir)
    print(len(all_arr), len(list_of_files))

    length = 47500
    timestep, features = 100, 4

    all_arr = same_length(all_arr, length) # all files same length
    print(len(all_arr), all_arr[0].shape)
    all_arr = reshape_data(all_arr, timestep, features) # reshape
    print(len(all_arr), all_arr[0].shape)
    cv1, cv2, cv3, cv4 = crossval_split(all_arr, list_of_files) # split into 4 folds based on trial/part_num

    # set up initial parameters
    output_directory = "./"
    #nb_classes = trainy.shape[1] 
    nb_classes = 11
    #input_shape = (trainX.shape[1], trainX.shape[2])
    input_shape = (timestep, features)
    ml_models = ["lr", "knn", "dt", "svm", "rf", "gb"] # to identify ml methods from dl
    classifier_name = str(sys.argv[2]) # determine what model to build

    # testing parameters
    num_epochs = 128 # [ 256, 128, 64]
    kernel_size = [2,3,4]
    mini_bs = [32, 64, 128, 256, 512, 1024]
    conv1d_filters = [8, 16, 32, 64, 128, 256]

    average = []

    # train models
    if classifier_name in ml_models:
        for fold in [cv1]:#, cv2, cv3, cv4]:
            trainX = fold[0]
            trainy = fold[1]
            testX = fold[2]
            testy = fold[3]
            # flatten
            trainX, testX = trainX.reshape(-1,timestep*features), testX.reshape(-1,timestep*features)
            classifier = create_ml_classifier(classifier_name)
            mean, std = test_train_trad(classifier, trainX, trainy, testX, testy)
            average.append(mean)
        print(sum(average)/len(average))
    else:
        # add tests
        for x in mini_bs:
            scores = [] # for storing cross validation results
            for fold in [cv1, cv2, cv3, cv4]:
                trainX = fold[0]
                trainy = fold[1]
                testX = fold[2]
                testy = fold[3]
                # encode labels
                trainy = to_categorical(trainy)
                testy = to_categorical(testy)
                print(trainX.shape, trainy.shape, testX.shape, testy.shape)
                classifier = create_classifier(classifier_name, input_shape, nb_classes, output_directory,x) 
                accuracy = classifier.fit(trainX, trainy, testX, testy, x)
                scores.append(accuracy)
            print("All scores: " + str(scores))
            print("Mean: " + str(np.mean(scores)*100) + " +/- " + str(np.std(scores)*100))
