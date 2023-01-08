# Gabriella Miles, Farscope PhD Student, Bristol Robotics Laboratory
# Date Created: 14th July 2020

import tensorflow.keras as keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv1D, Flatten, Dense
from datetime import datetime

class Classifier:
	def __init__(self, input_shape, num_classes, output_directory, variable, verbose=True, build=True):
		self.variable = variable
		self.output_directory = output_directory
		self.verbose = verbose

		if build:
			self.model = self.build_model(input_shape, num_classes)
			if self.verbose:
				self.model.summary()

	def build_model(self, input_shape, num_classes):
		input_layer = Input(input_shape)

		conv1 = Conv1D(filters=64, kernel_size=3, activation='relu')(input_layer)
		conv2 = Conv1D(filters=64, kernel_size=3, activation='relu')(conv1)
		flattened = Flatten()(conv2)
		
		dense_1 = Dense(100, activation='relu')(flattened)
		output_layer = Dense(num_classes, activation='softmax')(dense_1)

		model = Model(inputs=input_layer, outputs=output_layer)
		model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		return model

	def fit(self, trainX, trainy, testX, testy, variable):

		# hyperparameters
		mini_batch_size = 16
		num_epochs = 5
		suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
		log = "./logs/scalars/"+suffix
		my_callbacks = [
			tf.keras.callbacks.TensorBoard(log_dir=log), 
			tf.keras.callbacks.TerminateOnNaN(),
			tf.keras.callbacks.CSVLogger("./logs/test_"+suffix+".csv", separator=",", append=False)
		]

		hist =self.model.fit(trainX, trainy, validation_data=(testX,testy), callbacks=my_callbacks, batch_size=mini_batch_size, epochs=num_epochs, verbose=self.verbose)
		_, accuracy = self.model.evaluate(testX, testy,batch_size=mini_batch_size, verbose=self.verbose)
		if self.verbose:
			self.model.summary()

		return accuracy
