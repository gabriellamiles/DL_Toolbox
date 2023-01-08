# Gabriella Miles, Farscope PhD Student, Bristol Robotics Laboratory
# Created: 13th July 2020

import tensorflow.keras as keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Bidirectional, Flatten, Input, Dropout, LSTM, TimeDistributed
from datetime import datetime

class Classifier:
	def __init__(self, input_shape, num_classes, output_dir, variable, verbose=True, build=True):
		self.output_directory = output_dir
		print(type(input_shape), input_shape)
		self.variable = variable
		print(type(output_dir), output_dir)
		self.verbose = verbose

		if build:
			self.model = self.build_model(input_shape, num_classes)
			return

	def build_model(self, input_shape, num_classes):
		
		print(type(input_shape))	
		input_layer = Input(input_shape)

		lstm_1 = Bidirectional(LSTM(100, activation='relu'))(input_layer)
		lstm_1 = Dropout(0.5)(lstm_1)

		#dense_1 = TimeDistributed(Dense(100, activation='relu'))(lstm_1)
		output_layer = Dense(num_classes, activation='softmax')(lstm_1)

		model = Model(inputs=input_layer, outputs=output_layer)
		model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

		return model

	def fit(self, trainX, trainy, testX, testy, variable):

		# hyperparameters
		mini_batch_size = 16
		num_epochs = 30
		suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
		log = "./logs/scalars/"+suffix

		my_callbacks = [
			tf.keras.callbacks.TensorBoard(log_dir=log),
			tf.keras.callbacks.TerminateOnNaN(),
			tf.keras.callbacks.CSVLogger("./logs/test_"+suffix+".csv", separator=",", append=False)
		]

		# reshape labels for bidirectional lstm
		#trainy = trainy.reshape(trainy[0], 1, 11)

		hist =self.model.fit(trainX, trainy, validation_data=(testX, testy), callbacks=my_callbacks, batch_size=mini_batch_size, epochs=num_epochs, verbose=self.verbose)

		_, accuracy = self.model.evaluate(testX, testy,batch_size=mini_batch_size, verbose=self.verbose)
		#if self.verbose:
		#	self.model.summary()

		return accuracy
