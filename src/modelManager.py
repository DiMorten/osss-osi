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
from src.loss import categorical_focal
from sklearn.metrics import f1_score
class ModelManager():
	def __init__(self, pt):
		self.pt = pt

	def setArchitecture(self, modelArchitecture):
		self.model = modelArchitecture(img_shape = (self.pt.patch_size, self.pt.patch_size, 3), class_n = self.pt.class_n).build()

	def configure(self):
		loss=categorical_focal(alpha=0.25,gamma=2)
		metrics=['accuracy']

		optimizer = Adam(lr=self.pt.learning_rate, beta_1=0.9)

		self.model.compile(loss = loss,
			optimizer = optimizer,
			metrics = metrics)
	
	def loadWeights(self):
		self.model = load_model('best_model.h5', compile=False)

	def fit(self, trainGenerator, validationGenerator):

		es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
		mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
		callbacks = [es, mc]

		history = self.model.fit(trainGenerator,
			batch_size = self.pt.batch_size, 
			epochs = 70, 
			validation_data=validationGenerator,
			callbacks = callbacks,
			shuffle = False
			)
	
	def evaluate(self, ds):
		ds.predictions = self.infer(ds.X_test, ds.Y_test)
		# ds.predictions = self.model.predict(ds.testGenerator)
		ic(ds.predictions.shape, ds.Y_test.shape)

		f1 = f1_score(ds.Y_test.flatten(), ds.predictions.flatten(), average='macro')
		ic(f1)
		pdb.set_trace()

	def infer(self, X, Y, overlap_percent=0):
		num_ims, rows, cols, _ = X.shape
		# Percent of overlap between consecutive patches.
		overlap = round(self.pt.patch_size * overlap_percent)
		overlap -= overlap % 2
		stride = self.pt.patch_size - overlap
		# Add Padding to the image to match with the patch size
		step_row = (stride - rows % stride) % stride
		step_col = (stride - cols % stride) % stride
		pad_tuple_mask = ( (overlap//2, overlap//2 + step_row), ((overlap//2, overlap//2 + step_col)) )

		k1, k2 = (rows+step_row)//stride, (cols+step_col)//stride
		print('Total number of patches: %d x %d' %(k1, k2))
		
		predictions = np.zeros_like(Y)
		for im_id in range(num_ims):
			for i in range(k1):
				for j in range(k2):
					patch = X[im_id, i*stride:(i*stride + self.pt.patch_size), j*stride:(j*stride + self.pt.patch_size), :]
					patch = patch[np.newaxis,...]
					prediction = self.model.predict(patch)[0,...].argmax(axis=-1)
					# ic(prediction.shape)
					# pdb.set_trace()
					predictions[im_id, i*stride : i*stride+stride, 
                      			j*stride : j*stride+stride] = prediction[overlap//2 : overlap//2 + stride, 
																overlap//2 : overlap//2 + stride]
		return predictions