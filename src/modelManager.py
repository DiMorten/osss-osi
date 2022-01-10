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
	

	def fit(self, trainGenerator, validationGenerator):
		#callbacks = [MonitorGenerator(
		#		validation=validation_generator,
		#		patience=10, classes=self.pt.class_n)]
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