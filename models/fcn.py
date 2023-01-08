# Gabriella Miles, Farscope PhD, Bristol Robotics Laboratory
# Date created: 13th July 2020

import tensorflow.keras as keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Input, TimeDistributed, Conv1D, LSTM, MaxPooling1D, GlobalAveragePooling1D, Flatten, Dropout
from datetime import datetime

class Classifier:
    def __init__(self, input_shape, num_classes, output_directory, variable, verbose=True, build=True):
        self.output_directory = output_directory
        self.verbose = verbose
        self.logdir = "../logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_callback = keras.callbacks.TensorBoard(log_dir=self.logdir)

        if build == True:
            self.model = self.build_model(input_shape, num_classes)

            return

    def build_model(self, input_shape, num_classes):
        input_layer = Input(input_shape)
        conv_1 = Conv1D(filters=64, kernel_size=3, activation='relu')(input_layer)
        conv_2 = Conv1D(filters=64, kernel_size=3, activation='relu')(conv_1)
        #conv_2 = Dropout(0.5)(conv_2)
        conv_3 = Conv1D(filters=64, kernel_size=4, activation='relu')(conv_2)
        conv_3 = GlobalAveragePooling1D()(conv_3)
        output_layer = Dense(num_classes, activation='softmax')(conv_3)

        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    def fit(self, trainX, trainy, testX, testy, variable):
        mini_bs = variable 
        num_epochs = 128 

        hist = self.model.fit(trainX, trainy, batch_size=mini_bs, epochs=num_epochs, verbose=self.verbose, callbacks=[self.tensorboard_callback])

        _, accuracy = self.model.evaluate(testX, testy, batch_size=mini_bs, verbose=self.verbose)
        return accuracy
