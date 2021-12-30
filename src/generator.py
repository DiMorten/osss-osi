import cv2
import os
import glob
import numpy as np
import pdb
from pathlib import Path
import csv
import threading
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Dropout, Conv2DTranspose, ConvLSTM2D, Activation, BatchNormalization, Bidirectional, TimeDistributed, AveragePooling2D, MaxPooling2D, Lambda, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1,l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf


import sys
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import joblib

import tensorflow
from tensorflow.keras.applications.vgg16 import VGG16
from icecream import ic
import matplotlib.pyplot as plt
import pdb

class DataGeneratorWithCoords(keras.utils.Sequence):
	'Generates data for Keras'
	def __init__(self, inputs, labels, coords, samples_per_epoch = None,
				batch_size=16, dim=(1002, 250, 250), label_dim=(1002, 250, 250),
				n_channels=3, n_classes=4, shuffle=False, center_pixel = False, printCoords=False,
				augm = False):
		'Initialization'
		self.inputs = inputs
		self.dim = dim
		self.batch_size = batch_size
		ic(self.batch_size)
		self.patch_size = dim[1]
		ic(self.patch_size)

		self.labels = labels

		self.n_channels = n_channels
		self.n_classes = n_classes
		self.shuffle = shuffle
		self.label_dim = label_dim
		self.coords = coords
		self.center_pixel = center_pixel
		self.printCoords = printCoords
		self.augm = augm

		self.samples_per_epoch = samples_per_epoch
		self.on_epoch_end()

	def __len__(self):
		'Denotes the number of batches per epoch'
#		return int(np.floor(len(self.list_IDs) / self.batch_size))
		n_batches = int(np.floor(self.indexes.shape[0] / self.batch_size))
		ic(n_batches)
		return n_batches
	def __getitem__(self, index):
		'Generate one batch of data'
		# Generate indexes of the batch
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

		# Find list of IDs
#		list_IDs_temp = [self.list_IDs[k] for k in indexes]
		coords_batch = self.coords[indexes]


		# Generate data
		X, y = self.__data_generation(coords_batch)

		return X, y

	def on_epoch_end(self):
		'Updates indexes after each epoch'
#		self.indexes = np.arange(len(self.list_IDs))
		self.indexes = np.arange(self.coords.shape[0])

		if self.shuffle == True:
			np.random.shuffle(self.indexes)
		# self.indexes = self.indexes[:20000]
		#self.indexes = np.random.choice(self.indexes, 5000)
	def data_augmentation(self, X, Y):
		transf = np.random.randint(0,6,1)
		if transf == 0:
			# rot 90
			X = np.rot90(X,1,(0,1))
			Y = np.rot90(Y,1,(0,1))
			
		elif transf == 1:
			# rot 180
			X = np.rot90(X,2,(0,1))
			Y = np.rot90(Y,2,(0,1))
			
		elif transf == 2:
			# flip horizontal
			X = np.flip(X,0)
			Y = np.flip(Y,0)
			
		elif transf == 3:
			# flip vertical
			X = np.flip(X,1)
			Y = np.flip(Y,1)
			
		elif transf == 4:
			# rot 270
			X = np.rot90(X,3,(0,1))
			Y = np.rot90(Y,3,(0,1))
		return X, Y
	def __data_generation(self, coords_batch):
	
		'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
		# Initialization
		X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)
#		Y = np.empty((self.batch_size, *self.label_dim, self.n_classes), dtype=int)
		Y = np.empty((self.batch_size, *self.label_dim), dtype=int)

		if self.printCoords:
			ic(coords_batch)
		for idx in range(coords_batch.shape[0]):
			'''
			print(idx, coords_batch[idx], coords_batch[idx][0])
			print(coords_batch[idx][0]-self.patch_size//2)
			print(coords_batch[idx][0]+self.patch_size//2+self.patch_size%2)
			print(coords_batch[idx][1]-self.patch_size//2)
			print(coords_batch[idx][1]+self.patch_size//2+self.patch_size%2)
			ic(self.inputs.shape)
			ic(self.labels.shape)

			#pdb.set_trace()
			'''
			'''
			ic(self.inputs.shape, self.labels.shape)
			ic(self.center_pixel)
			ic(coords_batch[idx])
			ic(coords_batch[idx][1], 
				coords_batch[idx][1]+self.patch_size,
				coords_batch[idx][2],
				coords_batch[idx][2]+self.patch_size)
			'''
			if self.center_pixel == False:
				input_patch = self.inputs[coords_batch[idx][0], coords_batch[idx][1]:coords_batch[idx][1]+self.patch_size,
					coords_batch[idx][2]:coords_batch[idx][2]+self.patch_size]

				label_patch = self.labels[coords_batch[idx][0], coords_batch[idx][1]:coords_batch[idx][1]+self.patch_size,
					coords_batch[idx][2]:coords_batch[idx][2]+self.patch_size]
				# ic(input_patch.shape, label_patch.shape)

				
#				coords_batch[idx][0]-self.patch_size//2:coords_batch[idx][0]+self.patch_size//2+self.patch_size%2,
#						  coords_batch[idx][1]-self.patch_size//2:coords_batch[idx][1]+self.patch_size//2+self.patch_size%2]
##			ic(coords_batch[idx])
##			ic(label_patch)
##			pdb.set_trace()
			#ic(input_patch.shape)
			#ic(label_patch.shape)
			#pdb.set_trace()
			#ic(X.shape, Y.shape)
			#X, Y = self.data_augmentation(X, Y)
			#ic(X.shape, Y.shape)
#			self.augm = True
			
			if self.augm == True:
				transf = np.random.randint(0,6,1)
				if transf == 0:
					# rot 90
					input_patch = np.rot90(input_patch,1,(0,1))
					label_patch = np.rot90(label_patch,1,(0,1))
					
				elif transf == 1:
					# rot 180
					input_patch = np.rot90(input_patch,2,(0,1))
					label_patch = np.rot90(label_patch,2,(0,1))
					
				elif transf == 2:
					# flip horizontal
					input_patch = np.flip(input_patch,0)
					label_patch = np.flip(label_patch,0)
					
				elif transf == 3:
					# flip vertical
					input_patch = np.flip(input_patch,1)
					label_patch = np.flip(label_patch,1)
					
				elif transf == 4:
					# rot 270
					input_patch = np.rot90(input_patch,3,(0,1))
					label_patch = np.rot90(label_patch,3,(0,1))


			X[idx] = input_patch
#			Y[idx] = toOneHot(label_patch)
			Y[idx] = label_patch
		coords_print = False
		if coords_print == True:
			ic(coords_batch)
			ic(X.shape)
			ic(np.min(X), np.average(X), np.max(X))
			ic(Y.shape)
			ic(np.unique(Y, return_counts=True))
			pdb.set_trace()
		
		return X, np.expand_dims(Y, axis=-1)
