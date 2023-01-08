# Deep Learning Test Architecture for Eye Movement Dataset

Gabriella Miles, Farscope PhD Student, Bristol Robotics Laboratory

## How to run:

Call the program using ``python main.py``, and a number of command-line arguments, listed below.

|             | Inputs | Description                                                         |
| ----------- | ------ | ------------------------------------------------------------------- |
| sys.argv[1] |  str   | Filepath to location of data directory (all files stored together). |
| sys.argv[2] | model  | Name of model to test, options are listed below.                    |

## Models currently available for testing:

The table below defines the models that are available for testing, as well as the inputs for sys.argv[2] for each model. Where the model is given as **Machine Learning**, and the call sign is ``trad``, this asks the program to run a testing suite of traditional machine learning (not deep learning) algorithms. These models are: logistic regression, k-Nearest Neighbour, decision tree, support vector machines, random forest, and XGBoost (gradient boosting). 

| Model                     | Name    |
| ------------------------- | ------- |
| LSTM                      | lstm    |
| Reversed-LSTM             | rlstm   | 
| Bidirectional-LSTM        | blstm   |
| CNN-LSTM                  | cnnlstm |
| FCN                       | fcn     |
| T-LeNet                   | tlenet  |
| MLP                       | mlp     | 
| Logistic Regression       | lr      |
| kNN                       | knn     |
| Decision Tree             | dt      |
| Support Vector Machine    | svm     |
| Random Forest             | rf      |
| Gradient Boosting Machine | gb      |

## Default file structure (or where the data is kept):

``sys.argv[1]`` requires that the filepath to the directory where the data is stored be given. This is stored in a separate directory to the main testing harness, for ease of preprocessing. Assuming the default file structure configuration (see Figure X), is used, call the data using the filepath: ``../preprocessing_useful/data/synch_data``.

