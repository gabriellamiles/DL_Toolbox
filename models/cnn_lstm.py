# Gabriella Miles, Farscope PhD Student, Bristol Robotics Laboratory
# Date created: 13th July 2020

import tensorflow.keras as keras
import tensorflow as tf
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Conv1D, LSTM, Dense, TimeDistributed, Flatten, Dropout, MaxPooling1D
from datetime import datetime

class Classifier:
    def __init__(self, input_shape, num_classes, output_directory, variable, verbose=False, build=True):
        self.output_directory = output_directory
        self.verbose = verbose
        self.variable = variable

        # key parameters for cnn_lstm model
        self.cnn_subseq = 2
        self.cnn_steps = int(input_shape[0]/self.cnn_subseq)
        self.cnn_feats = input_shape[1]

        # rejig input shape to timedistributed conv layers
        cnn_input = (None, self.cnn_steps, self.cnn_feats)

        if build:
            self.model = self.build_model(cnn_input, num_classes, variable)
        if self.verbose:
            self.model.summary()

    def build_model(self, input_shape, num_classes, variable):
        # tuning the optimiser at the moment (adam)
        opt = Adam(lr= 0.0001, beta_1=0.7)

        input_layer = Input(input_shape)
        conv_1 = TimeDistributed(Conv1D(filters=variable, kernel_size=3, activation='relu'))(input_layer)
        conv_2 = TimeDistributed(Conv1D(filters=variable, kernel_size=3, activation='relu'))(conv_1)
        conv_2 = Dropout(0.5)(conv_2)
        conv_2 = TimeDistributed(MaxPooling1D(pool_size=2))(conv_2)

        flatten = TimeDistributed(Flatten())(conv_2)
        lstm_1 = LSTM(100, activation='relu')(flatten)
        lstm_1 = Dropout(0.5)(lstm_1)

        dense_1 = Dense(100, activation='relu')(lstm_1)
        output_layer = Dense(num_classes, activation='softmax')(dense_1)

        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        return model

    def fit(self, trainX, trainy, testX, testy, variable):

        # hyperparameters
        mini_batch_size = 16
        num_epochs = 30
        suffix = datetime.now().strftime("%Y%m%d_%H%M%D")
        log = "./logs/scalars/"+suffix

        my_callbacks = [
            tf.keras.callbacks.TensorBoard(log_dir=log),
            tf.keras.callbacks.TerminateOnNaN(),
            tf.keras.callbacks.CSVLogger("./logs/test_"+suffix+".csv", separator=",", append=False)
        ]

        # reshape data to make correct for cnn
        trainX = trainX.reshape(trainX.shape[0], self.cnn_subseq, self.cnn_steps, self.cnn_feats)
        testX = testX.reshape(testX.shape[0], self.cnn_subseq, self.cnn_steps, self.cnn_feats)

        hist =self.model.fit(trainX, trainy, validation_data=(testX, testy), batch_size=mini_batch_size, callbacks=my_callbacks, epochs=num_epochs, verbose=self.verbose)
        _, accuracy = self.model.evaluate(testX, testy,batch_size=mini_batch_size, verbose=self.verbose)

        return accuracy
