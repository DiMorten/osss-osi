# from utils_v1 import *
import os
import tensorflow as tf
from tensorflow.keras.layers import *
#from skimage.util.shape import view_as_windows
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow.keras.backend as K
import numpy as np
import matplotlib.pyplot as plt

from icecream import ic
import cv2
import pathlib
import time
from sklearn import metrics
import glob
import cv2
tf.random.set_seed(2)
np.random.seed(2)

t0 = time.time()

import os
os.getcwd()
import pdb
import pickle
from src.generator import DataGeneratorWithCoords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from collections import Counter

def extract_coords(num_ims, rows, cols, patch_size, overlap_percent=0,
	ignore_only_class0_patches = False, label = None):

	"""
	Everything  in this function is made operating with
	the upper left corner of the patch
	"""


	# Percent of overlap between consecutive patches.
	overlap = round(patch_size * overlap_percent)
	overlap -= overlap % 2
	stride = patch_size - overlap
	# Add Padding to the image to match with the patch size
	step_row = (stride - rows % stride) % stride
	step_col = (stride - cols % stride) % stride
	pad_tuple_mask = ( (overlap//2, overlap//2 + step_row), ((overlap//2, overlap//2 + step_col)) )
#    lbl = np.pad(lbl, pad_tuple_mask, mode = 'symmetric')

	k1, k2 = (rows+step_row)//stride, (cols+step_col)//stride
	print('Total number of patches: %d x %d' %(k1, k2))

	patch_coords = []
	for im_id in range(num_ims):
		for i in range(k1):
			for j in range(k2):
				if ignore_only_class0_patches == True:
					label_patch = label[im_id, i*stride:(i+1)*stride, j*stride:(j+1)*stride]
					if np.all(label_patch == 0): continue
				patch_coords.append((im_id, i*stride, j*stride))
	print("Patch coords ",len(patch_coords))
	return np.asarray(patch_coords), step_row, step_col, overlap, pad_tuple_mask

class Dataset():
	def __init__(self, pt):
		self.pt = pt

	def loadSet(self, folderName):
		ic(self.pt.dataPath / folderName)
		x_list = glob.glob(str(self.pt.dataPath / folderName / "images") + "/*")
		y_list = glob.glob(str(self.pt.dataPath / folderName / "labels_1D") + "/*")
		X = []
		Y = []
		for idx, (x_name, y_name) in enumerate(zip(x_list, y_list)):
			x = cv2.imread(x_name)
			y = cv2.imread(y_name)
			#ic(x.shape, y.shape)
			X.append(x)
			Y.append(y)
#            pdb.set_trace()
		X_np = np.expand_dims(np.stack(X, axis = 0)[...,0], axis = -1)
		Y_np = np.stack(Y, axis = 0)[...,0]
		return X_np.astype(np.float16), Y_np.astype(np.float16) 
	def load(self):
		self.X_train, self.Y_train = self.loadSet("train")
		self.X_test, self.Y_test = self.loadSet("test")
	
		ic(self.X_train.shape, self.Y_train.shape, self.X_test.shape, self.Y_test.shape)
		ic(self.X_train.dtype, self.Y_train.dtype, self.X_test.dtype, self.Y_test.dtype)
		#pdb.set_trace()
		ic(np.average(self.X_train))
		# pdb.set_trace()

	def channelFlatten(self, x):
		shape = x.shape
		return np.reshape(x, (-1, shape[-1])), shape
	
	def channelUnflatten(self, x, shape):
		return np.reshape(x, shape)
		
	def normalize(self):
		self.X_train, X_train_shape = self.channelFlatten(self.X_train)
		self.X_test, X_test_shape = self.channelFlatten(self.X_test)
		ic(self.X_train.shape, self.X_test.shape)

		self.scaler = MinMaxScaler()
		self.X_train = self.scaler.fit_transform(self.X_train)
		self.X_test = self.scaler.transform(self.X_test)

		self.X_train = self.channelUnflatten(self.X_train, X_train_shape)
		self.X_test = self.channelUnflatten(self.X_test, X_test_shape)
		ic(self.X_train.shape, self.X_test.shape)


	def extractCoords(self, num_ims = 1002, rows = 650, cols = 1250, patch_size = 250, overlap_percent=0):


		"""
		Everything  in this function is made operating with
		the upper left corner of the patch
		"""
		loadCoords = False
		if loadCoords == False:
			self.coords_train, step_row, step_col, overlap, self.pad_tuple_mask = extract_coords(
				self.X_train.shape[0], self.pt.h, self.pt.w, self.pt.patch_size, overlap_percent=0,
				ignore_only_class0_patches = False)

			ic(self.coords_train.shape, step_row, step_col, overlap, self.pad_tuple_mask)

			self.coords_validation, step_row, step_col, overlap, self.pad_tuple_mask = extract_coords(
				self.X_validation.shape[0], self.pt.h, self.pt.w, self.pt.patch_size, overlap_percent=0)

			ic(self.coords_validation.shape, step_row, step_col, overlap, self.pad_tuple_mask)

			self.coords_test, step_row, step_col, overlap, self.pad_tuple_mask = extract_coords(
				self.X_test.shape[0], self.pt.h, self.pt.w, self.pt.patch_size, overlap_percent=0)

			ic(self.coords_test.shape, step_row, step_col, overlap, self.pad_tuple_mask)

			np.savez('coords.npz', coords_train = self.coords_train,
				coords_validation = self.coords_validation,
				coords_test = self.coords_test)
			with open('pad_tuple_mask.pickle', 'wb') as f:
				pickle.dump(self.pad_tuple_mask, f)
		else:
			data = np.load('coords.npz')
			self.coords_train = data['coords_train']
			self.coords_validation = data['coords_validation']
			self.coords_test = data['coords_test']
			with open('data.pickle', 'rb') as f:
				self.pad_tuple_mask = pickle.load(f)

	def addPadding(self):
		pad_tuple = ((0, 0),) + self.pad_tuple_mask + ((0, 0),) 
		Y_pad_tuple = ((0, 0),) + self.pad_tuple_mask

		ic(pad_tuple)
		ic(self.X_train.shape)
		self.X_train = np.pad(self.X_train, pad_tuple, mode = 'symmetric')
		ic(self.Y_train.shape)
		self.Y_train = np.pad(self.Y_train, Y_pad_tuple, mode = 'symmetric')

		self.X_validation = np.pad(self.X_validation, pad_tuple, mode = 'symmetric')
		ic(self.Y_validation.shape)
		self.Y_validation = np.pad(self.Y_validation, Y_pad_tuple, mode = 'symmetric')

		self.X_test = np.pad(self.X_test, pad_tuple, mode = 'symmetric')
		ic(self.Y_test.shape)
		self.Y_test = np.pad(self.Y_test, Y_pad_tuple, mode = 'symmetric')

	def removePadding(self, x):
		ic(self.pad_tuple_mask)
		ic(x.shape)
		x = x[:, :-self.pad_tuple_mask[0][1], :-self.pad_tuple_mask[1][1]]
		# ic(x.shape)
		# pdb.set_trace()

		return x
	def removePadding(self, x):
		x = x[:, :-6, :-14]
		return x
	def toOneHot(self, label, class_n):

		# convert to one-hot
		label_shape = label.shape
		ic(label[0])
		ic(label_shape)
#				ic((*label_shape[:-1], -1))
		label = np.reshape(label, -1)
		b = np.zeros((label.shape[0], class_n))
		b[np.arange(label.shape[0]), label] = 1
		ic(b.shape)

		#b = np.reshape(b, label_shape)
		ic(b.shape)
		#ic(b[0])
		#pdb.set_trace()

		return b
		
	def setTrainGenerator(self, num_ims, rows, cols, patch_size):
		params_train = {
			'dim': (patch_size,patch_size),
			'label_dim': (patch_size,patch_size),
			'batch_size': self.pt.batch_size,
			'n_classes': self.pt.class_n, # it was 6. Now it is 13 + 1 = 14
			'n_channels': self.pt.channel_n,
			'shuffle': False,
			'augm': False,
			'subsample': False}        
		
		ic(self.X_train.shape, self.Y_train.shape)
		ic(np.unique(self.Y_train, return_counts=True))

		self.trainGenerator = DataGeneratorWithCoords(self.X_train, self.Y_train, 
					self.coords_train, **params_train)

		params_validation = params_train.copy()
		params_validation['shuffle'] = False
		params_validation['augm'] = False
		params_validation['subsample'] = False

		self.validationGenerator = DataGeneratorWithCoords(self.X_validation, self.Y_validation, 
					self.coords_validation, **params_validation)

		params_test = params_validation.copy()
		self.testGenerator = DataGeneratorWithCoords(self.X_test, self.Y_test, 
					self.coords_test, **params_test)


	def trainValSplit(self, validation_split=0.15):
		idxs = range(self.pt.num_ims_train)
		self.X_train, self.X_validation, self.Y_train, self.Y_validation = train_test_split(
				self.X_train, self.Y_train, test_size = validation_split)
		self.num_ims_train = self.X_train.shape[0]
		ic(self.X_train.shape, self.X_validation.shape, self.Y_train.shape, self.Y_validation.shape)
		
	def trainReduce(self, trainSize = 20):
		len_coords_train = len(self.coords_train)
		idxs = np.random.choice(len_coords_train, trainSize)
		self.coords_train = self.coords_train[idxs]
	
	def useLabelsAsInput(self):
		self.X_train = np.expand_dims(self.Y_train.copy().astype(np.float16), axis = -1)
	
	def balance(self, samples_per_class=500):
		patch_count=np.zeros(self.pt.class_n)
		patch_count_axis = (1,2)
		rotation_axis = (1,2)

		balance={}
		balance["out_n"]=(self.pt.class_n)*samples_per_class
		ic(self.coords_train.shape)
		balance["coords"] = np.zeros((balance["out_n"], *self.coords_train.shape[1:])).astype(np.int)
		ic(balance["coords"].shape)

		coords_classes = np.zeros((self.coords_train.shape[0], self.pt.class_n))
		ic(coords_classes.shape)
		unique_train = np.unique(self.Y_train)
		ic(unique_train)

		for idx, coord in enumerate(self.coords_train):
			label_patch = self.Y_train[coord[0], coord[1]:coord[1]+self.pt.patch_size,
				coord[2]:coord[2]+self.pt.patch_size]
			patchClassCount = Counter(label_patch.flatten()) # es mayor a 0? o el bcknd esta al final?
			for key in patchClassCount:
				key = int(key)
# 				ic(key, patch_count.shape)
# 				pdb.set_trace()
				patch_count[key] = patch_count[key] + 1
				coords_classes[idx, key] = 1
		
		ic(patch_count)
		for k, clss in enumerate(range(self.pt.class_n)):
			ic(patch_count[clss])

			if patch_count[clss]==0:
				continue
			ic(clss)

			idxs = coords_classes[:, clss] == 1 # bool telling which samples belong to this class
			ic(idxs.shape,idxs.dtype)
			ic(np.unique(idxs, return_counts = True))
			# pdb.set_trace()

			balance["class_coords"]=self.coords_train[idxs]

			ic(balance["class_coords"].shape)
			ic(samples_per_class)
			ic(clss)

			if balance["class_coords"].shape[0]>samples_per_class:
				replace=False
			else:
				replace=True
#					index = range(balance["label"].shape[0])
			index = range(balance["class_coords"].shape[0])

			index = np.random.choice(index, samples_per_class, replace=replace)
			balance["coords"][k*samples_per_class:k*samples_per_class + samples_per_class] = balance["class_coords"][index]


		idx = np.random.permutation(balance["coords"].shape[0])

		self.coords_train = balance["coords"][idx]
		print("Balanced train unique (coords):")
		ic(self.coords_train.shape)		

	def getPatchesFromCoords(self, ims, coords):
		patch_size = self.pt.patch_size #32
		# t_len = 12
		# channel_n = 2
		patch_dim = (patch_size, patch_size)
		Y = np.empty((coords.shape[0], *patch_dim), dtype=np.float16)
		for idx, coord in enumerate(coords):
			patch = ims[coord[0], 
						coord[1]:coord[1]+patch_size,
						coord[2]:coord[2]+patch_size]
			Y[idx] = patch
		return Y

	def addPaddingInference(self):
		mask_pad = ((0,0), (0,6), (0,14), (0,0))
		mask_pad_label = ((0,0), (0,6), (0,14))

		self.X_test = np.pad(self.X_test, mask_pad, mode = 'symmetric')
		self.Y_test = np.pad(self.Y_test, mask_pad_label, mode = 'symmetric')
		ic(self.X_test.shape)
		pdb.set_trace()

		