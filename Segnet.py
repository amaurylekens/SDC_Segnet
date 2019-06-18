import tensorflow as tf
import numpy as np

from keras import models
from keras.layers.core import Activation, Reshape, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD

class Segnet:
	def __init__(self):

		img_w = 256
		img_h = 256
		n_labels = 2

		kernel = 3
		self.nn = models.Sequential()
		
		# fist encoding part
		self.nn.add(Convolution2D(64, kernel, border_mode='same', input_shape=(img_h, img_w,3)))
		self.nn.add(BatchNormalization())
		self.nn.add(Activation('relu'))
		self.nn.add(Convolution2D(64, kernel, border_mode='same'))
		self.nn.add(BatchNormalization())
		self.nn.add(Activation('relu'))
		self.nn.add(MaxPooling2D())

		# second encoding part
		self.nn.add(Convolution2D(128, kernel, kernel, border_mode='same'))
		self.nn.add(BatchNormalization())
		self.nn.add(Activation('relu'))
		self.nn.add(Convolution2D(128, kernel, kernel, border_mode='same'))
		self.nn.add(BatchNormalization())
		self.nn.add(Activation('relu'))
		self.nn.add(MaxPooling2D())

		# third encoding part
		self.nn.add(Convolution2D(256, kernel, kernel, border_mode='same'))
		self.nn.add(BatchNormalization())
		self.nn.add(Activation('relu'))
		self.nn.add(Convolution2D(256, kernel, kernel, border_mode='same'))
		self.nn.add(BatchNormalization())
		self.nn.add(Activation('relu'))
		self.nn.add(Convolution2D(256, kernel, kernel, border_mode='same'))
		self.nn.add(BatchNormalization())
		self.nn.add(Activation('relu'))
		self.nn.add(MaxPooling2D())

		# fourth encoding part
		self.nn.add(Convolution2D(256, kernel, kernel, border_mode='same'))
		self.nn.add(BatchNormalization())
		self.nn.add(Activation('relu'))
		self.nn.add(Convolution2D(256, kernel, kernel, border_mode='same'))
		self.nn.add(BatchNormalization())
		self.nn.add(Activation('relu'))
		self.nn.add(Convolution2D(256, kernel, kernel, border_mode='same'))
		self.nn.add(BatchNormalization())
		self.nn.add(Activation('relu'))
		self.nn.add(MaxPooling2D())

		# fifth encoding part
		self.nn.add(Convolution2D(256, kernel, kernel, border_mode='same'))
		self.nn.add(BatchNormalization())
		self.nn.add(Activation('relu'))
		self.nn.add(Convolution2D(256, kernel, kernel, border_mode='same'))
		self.nn.add(BatchNormalization())
		self.nn.add(Activation('relu'))
		self.nn.add(Convolution2D(256, kernel, kernel, border_mode='same'))
		self.nn.add(BatchNormalization())
		self.nn.add(Activation('relu'))
		self.nn.add(MaxPooling2D())


		# first decoding part
		self.nn.add(UpSampling2D())
		self.nn.add(Convolution2D(512, kernel, kernel, border_mode='same'))
		self.nn.add(BatchNormalization())
		self.nn.add(Activation('relu'))
		self.nn.add(Convolution2D(512, kernel, kernel, border_mode='same'))
		self.nn.add(BatchNormalization())
		self.nn.add(Activation('relu'))
		self.nn.add(Convolution2D(512, kernel, kernel, border_mode='same'))
		self.nn.add(BatchNormalization())
		self.nn.add(Activation('relu'))

		# second decoding part
		self.nn.add(UpSampling2D())
		self.nn.add(Convolution2D(512, kernel, kernel, border_mode='same'))
		self.nn.add(BatchNormalization())
		self.nn.add(Activation('relu'))
		self.nn.add(Convolution2D(512, kernel, kernel, border_mode='same'))
		self.nn.add(BatchNormalization())
		self.nn.add(Activation('relu'))
		self.nn.add(Convolution2D(512, kernel, kernel, border_mode='same'))
		self.nn.add(BatchNormalization())
		self.nn.add(Activation('relu'))

		# third decoding part
		self.nn.add(UpSampling2D())
		self.nn.add(Convolution2D(256, kernel, kernel, border_mode='same'))
		self.nn.add(BatchNormalization())
		self.nn.add(Activation('relu'))
		self.nn.add(Convolution2D(256, kernel, kernel, border_mode='same'))
		self.nn.add(BatchNormalization())
		self.nn.add(Activation('relu'))
		self.nn.add(Convolution2D(128, kernel, kernel, border_mode='same'))
		self.nn.add(BatchNormalization())
		self.nn.add(Activation('relu'))

		# fourth decoding part
		self.nn.add(UpSampling2D())
		self.nn.add(Convolution2D(128, kernel, kernel, border_mode='same'))
		self.nn.add(BatchNormalization())
		self.nn.add(Activation('relu'))
		self.nn.add(Convolution2D(64, kernel, kernel, border_mode='same'))
		self.nn.add(BatchNormalization())
		self.nn.add(Activation('relu'))

		# fifth decoding part
		self.nn.add(UpSampling2D())
		self.nn.add(Convolution2D(64, kernel, kernel, border_mode='same'))
		self.nn.add(BatchNormalization())
		self.nn.add(Activation('relu'))
		self.nn.add(Convolution2D(n_labels, kernel, kernel, border_mode='same'))
		self.nn.add(BatchNormalization())

		self.nn.add(Reshape((n_labels, img_h*img_w)))
		self.nn.add(Permute((2,1)))
		self.nn.add(Activation('softmax'))

	def train(self, train_data, train_label, nb_epoch, batch_size):

		optimizer = SGD(lr=0.001, momentum=0.9, decay=0.0005, nesterov=False)
		self.nn.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
		print("the model is compiled")

		nb_epoch = 50
		batch_size = 18
		history = self.nn.fit(train_data, train_label, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1)
		self.nn.save_weights('model_weight.hdf5')

	def predict(self, data):
		output = self.nn.predict_proba(data, verbose=0)

		return output

	def load_weight(self):
		self.nn.load_weights('model_weight.hdf5')
















































