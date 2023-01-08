# Gabriella Miles, Farscope PhD Student, Bristol Robotics Laboratory
# Created: 9th July 2020

import keras
import tensorflow as tf
import time
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input, Flatten
from datetime import datetime

class Classifier:
	def __init__(self, input_shape, nb_classes, output_dir, variable, verbose=True, build=True):
		"""
		Docstring
		"""
		self.output_dir = output_dir
		self.verbose = verbose
		self.variable = variable

		if build:
			self.model = self.build_model_sequential(input_shape, nb_classes)
			if self.verbose:
				self.model.summary()
			return

	def build_model_sequential(self, input_shape, nb_classes):
		""" 
		Docstring
		"""

		input_layer = Input(input_shape)

		lstm_1 = LSTM(100, return_sequences=True, activation='relu')(input_layer)
		lstm_2 = LSTM(100, return_sequences=True, activation='relu')(lstm_1)
		lstm_3 = LSTM(100, activation='relu')(lstm_2)

		output_layer = Dense(nb_classes, activation='softmax')(lstm_3)
		model = Model(inputs=input_layer, outputs=output_layer)

		#model = Sequential()

		#model.add(LSTM(100,input_shape=input_shape,return_sequences=True, activation='relu'))
		#model.add(LSTM(100, return_sequences=True, activation='relu'))
		#model.add(LSTM(100, activation='relu'))
		#model.add(Dense(nb_classes, activation='softmax'))
		model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

		return model


	def fit(self, trainX, trainy, testX, testy, variable):

		#hyperparameters
		mini_batch_size = 16
		nb_epochs = 5
		suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
		log = "./logs/scalars/"+suffix
		my_callbacks = [
			tf.keras.callbacks.TensorBoard(log_dir=log),
			tf.keras.callbacks.TerminateOnNaN(),
			tf.keras.callbacks.CSVLogger("./logs/test_"+suffix+".csv", separator=",", append=False)
		]

		start_time = time.time()
		hist = self.model.fit(trainX, trainy, validation_data=(testX,testy), callbacks=my_callbacks, batch_size=mini_batch_size, epochs=nb_epochs)

		_, accuracy = self.model.evaluate(testX, testy, batch_size=mini_batch_size, verbose=self.verbose)

		return accuracy
