# Gabriella Miles, Farscope PhD Student, Bristol Robotics Laboratory
# Created: 13th July 2020

import tensorflow.keras as keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from datetime import datetime

class Classifier:
	def __init__(self, input_shape, num_classes, output_directory, variable, verbose=True, build=True):
		self.output_directory = output_directory
		self.verbose = verbose
		self.variable = variable

		if build:
			self.model = self.build_model(input_shape, num_classes)
			return

	def build_model(self, input_shape, num_classes):
		input_layer = Input(input_shape)
		flatten_1 = Flatten()(input_layer)
		dense_1 = Dense(100, activation='relu')(flatten_1)
		dense_2 = Dense(100, activation='relu')(dense_1)
		output_layer = Dense(num_classes, activation='softmax')(dense_2)

		model = Model(inputs=input_layer, outputs=output_layer)

		model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

		return model

	def fit(self, trainX, trainy, testX, testy, variable):
		# hyperparameters
		mini_bs = 16
		num_epochs = 13 
		suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
		log = "./logs/scalars/"+suffix
		my_callbacks = [
			tf.keras.callbacks.TensorBoard(log_dir=log),
			tf.keras.callbacks.TerminateOnNaN(),
			tf.keras.callbacks.CSVLogger("./logs/test_"+suffix+".csv". separator=",", append=False)
		]
	
		hist = self.model.fit(trainX, trainy, validation_data=(testX, testy), callbacks=my_callbacks, batch_size=mini_bs, epochs=num_epochs, verbose=self.verbose)
		_, accuracy = self.model.evaluate(testX, testy, batch_size=mini_bs, verbose=self.verbose)

		if self.verbose:
			self.model.summary()

		return accuracy
