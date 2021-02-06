import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.regularizers import l2
import keras


class NeuralNetwork():

	"""A class for build a multilayer perceptron"""

	def __init__(self):
		self.model = Sequential()
		self.memory_x = []
		self.memory_y = []
		self.l2 = 0.001
		self.lr = 0.001


	def load_model(self, filepath):
		"""
		Save actual model
		:param filepath: path where model is going to be saved
		:return: nothing
		"""
		print('[Model] Loading model from file %s' % filepath)
		self.model = load_model(filepath)

	def build_model(self, input_shape):
		"""
		Build the multilayer perceptron model
		:param input_shape:
		:return: nothing
		"""

		self.model.add(Dense(128, activation='relu', input_shape = input_shape, kernel_regularizer=l2(self.l2), bias_regularizer=l2(self.l2)))
		#self.model.add(Dropout(0.5))
		self.model.add(Dense(64, activation='relu', kernel_regularizer=l2(self.l2), bias_regularizer=l2(self.l2)))
		#self.model.add(Dropout(0.5))
		self.model.add(Dense(16, activation='relu', kernel_regularizer=l2(self.l2), bias_regularizer=l2(self.l2)))
		#self.model.add(Dropout(0.5))
		self.model.add(Dense(8, activation='relu', kernel_regularizer=l2(self.l2), bias_regularizer=l2(self.l2)))
		#self.model.add(Dropout(0.5))
		self.model.add(Flatten())
		self.model.add(Dense(1, activation='tanh', kernel_regularizer=l2(self.l2), bias_regularizer=l2(self.l2)))

		sgd = SGD(lr=self.lr, decay=1e-6, momentum=0.5, nesterov=True)
		self.model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
		#self.model.compile(optimizer = keras.optimizers.Adam(lr = 1e-4, decay=1e-6),
		#	              loss = keras.losses.binary_crossentropy,
		#	              metrics = ['accuracy'])

	def train(self, x, y, epochs, verbose=0):
		"""
		Method to train the model
		:param x: input for the model
		:param y: desired output
		:param epochs: number of epochs to train
		:param verbose: param to indicate if train process is displayed
		:return:
		"""

		self.model.fit(x, y, epochs=epochs, verbose=verbose)

	def predict(self, data):
		"""
		Get output from the model for data
		:param data:
		:return: model prediction
		"""
		return self.model.predict(np.array([data]))

	def get_accuracy(self, X, y):
		"""
		Get the model accuracy for X data
		:param X: data to make the predictions
		:param y: real labels of X
		:return: predictions accuracy
		"""

		y_pred = self.model.predict(X)
		y_pred = (y_pred > 0.5)
		correct = 0

		for i in range(len(y_pred)):
			if y_pred[i] == y[i]:
				correct += 1

		accuracy = correct/len(y_pred)*100.0

		return accuracy

	def init_memory(self, X, y):
		"""
		Method to initialize the memory used for reforce learning
		:param X: Array with the data to init de memory
		:param y: Array with the labels of the X array
		:return: nothing
		"""
		for i in range(len(X)):
			self.memory_x.append(X[i])
			self.memory_y.append(y[i])

	def update_memory(self, data, target):
		"""
		Update memory for the reforce learning
		:param data: new data to train
		:param target: data label
		:return: nothing
		"""
		self.memory_x.insert(0, data)
		self.memory_x.pop()

		self.memory_y.insert(0, target)
		self.memory_y.pop()

	def reTrain(self, batch_size=None):
		"""
		ReTrain the model with data in memory
		:param batch_size: the size of the batch
		:return: nothing
		"""

		if batch_size == None:
			batch_size = len(np.array(self.memory_x))

		X = np.array(self.memory_x)
		y = np.array(self.memory_y)

		self.model.fit(X, y, epochs=5, batch_size=batch_size, verbose=0)
