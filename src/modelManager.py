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
from src.loss import categorical_focal, weighted_categorical_crossentropy, weighted_categorical_focal, categorical_crossentropy
from src.monitor import Monitor
from sklearn.metrics import f1_score, accuracy_score, jaccard_score
from sklearn.utils.class_weight import compute_class_weight
from icecream import ic

def plot_history(model, feature, path_file = None):

	val = "val_" + feature
	
	plt.xlabel('Epoch Number')
	plt.ylabel(feature)
	plt.plot(model.history[feature])
	plt.plot(model.history[val])
	plt.legend(["train_"+feature, val])    
	if path_file:
		plt.savefig(path_file)
		
class ModelManager():
	def __init__(self, pt, modelId = ""):
		self.pt = pt
		self.model_name = "results/best_model" + self.pt.modelId + ".h5"
	def setArchitecture(self, modelArchitecture):
		self.model = modelArchitecture(img_shape = (self.pt.patch_h, self.pt.patch_w, self.pt.channel_n), class_n = self.pt.class_n).build()
		ic(self.model)
	def computeWeights(self, y):
		unique, count = np.unique(y, return_counts=True) 
		self.weights = np.max(count) / count
		#self.weights = compute_class_weight("balanced", np.unique(y), y)
		self.weights = self.weights.astype(np.float16)
		ic(np.unique(y, return_counts=True))
		ic(self.weights)

	def configure(self):
		if self.pt.loss == "categorical_focal":
			loss=categorical_focal(alpha=0.25,gamma=2)
			# self.weights = np.asarray([1,1,1,1,1])
			# self.weights = np.asarray([1,22,3.6,50,3.6])
			
			# loss=weighted_categorical_focal(
			# 	self.weights, 
			# 	alpha=0.25,gamma=2)
			
		elif self.pt.loss == "weighted_categorical_crossentropy":
			loss = weighted_categorical_crossentropy(self.weights)
		elif self.pt.loss == "categorical_crossentropy":
			loss = categorical_crossentropy()
		metrics=['accuracy']

		optimizer = Adam(lr=self.pt.learning_rate, beta_1=0.9)

		self.model.compile(loss = loss,
			optimizer = optimizer,
			metrics = metrics)
	
	def loadWeights(self):
		ic(self.model_name)
		self.model.load_weights(self.model_name)
#		self.model = load_model(self.model_name, compile=False)
		ic(self.model.summary())

	def fit(self, trainGenerator, validationGenerator):
		lr_reduce = ReduceLROnPlateau(factor=0.9, min_delta=0.0001, patience=5, verbose=1)
		es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
		mc = ModelCheckpoint(self.model_name, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
		monitor = Monitor(validationGenerator, self.pt.class_n)
		# callbacks = [es, mc, monitor]
		callbacks = [es, mc]

		history = self.model.fit(trainGenerator,
			batch_size = self.pt.batch_size, 
			epochs = 200, 
			validation_data=validationGenerator,
			callbacks = callbacks,
			shuffle = False
			)

		plot_history(history, 'loss', path_file='loss_fig.png')
	
	def evaluate(self, ds):
		ds.predictions = self.infer(ds.X_test, ds.Y_test)
		ds.predictions = ds.removePadding(ds.predictions)
		ds.Y_test = ds.removePadding(ds.Y_test)

		# ds.predictions = self.model.predict(ds.testGenerator)
		ic(ds.predictions.shape, ds.Y_test.shape)

		f1_avg = f1_score(ds.Y_test.flatten(), ds.predictions.flatten(), average='macro')
		f1 = f1_score(ds.Y_test.flatten(), ds.predictions.flatten(), average=None)
		oa = accuracy_score(ds.Y_test.flatten(), ds.predictions.flatten())
		jaccard = jaccard_score(ds.Y_test.flatten(), ds.predictions.flatten(), average='macro')
		ic(f1_avg, f1, oa, jaccard)

		qualitative_results_id = [1, 81, 32, 45, 42, 81]
		for id_ in qualitative_results_id:
			cv2.imwrite('predictions_'+str(id_)+'.png',ds.predictions[id_].astype(np.float32)*50)
			cv2.imwrite('label_'+str(id_)+'.png',ds.Y_test[id_].astype(np.float32)*50)

	def evaluateEnsemble(self, ds, times = 10):
		num_ims_test, h, w = ds.Y_test.shape
		
		ds.predictions_ensemble = np.ones((times, num_ims_test,h,w, self.pt.class_n), dtype = np.float16)
		for t in range(times):
			ic(t)
			_, prediction_probabilities = self.infer(ds.X_test, ds.Y_test, save_prediction_probability = True)
			ds.predictions_ensemble[t] = prediction_probabilities.copy()

		return ds

	def infer(self, X, Y, overlap_percent=0, save_prediction_probability = False):
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
		prediction_probabilities = np.zeros((Y.shape[0], Y.shape[1], Y.shape[2], self.pt.class_n), dtype = np.float16)
		for im_id in range(num_ims):
			for i in range(k1):
				for j in range(k2):
					patch = X[im_id, i*stride:(i*stride + self.pt.patch_size), j*stride:(j*stride + self.pt.patch_size), :]
					patch = patch[np.newaxis,...]
					prediction_probability = self.model.predict(patch)[0,...]
					prediction = prediction_probability.argmax(axis=-1)
					# ic(prediction.shape)
					# pdb.set_trace()
					predictions[im_id, i*stride : i*stride+stride, 
                      			j*stride : j*stride+stride] = prediction[overlap//2 : overlap//2 + stride, 
																overlap//2 : overlap//2 + stride]
					if save_prediction_probability == True:
						prediction_probabilities[im_id, i*stride : i*stride+stride, 
									j*stride : j*stride+stride] = prediction_probability[overlap//2 : overlap//2 + stride, 
																	overlap//2 : overlap//2 + stride]

		if save_prediction_probability == True:		
			return predictions, prediction_probabilities
		return predictions

	def infer(self, X, Y, overlap_percent=0, save_prediction_probability = False):
		ic(self.model.summary())
		# Y = Y[]
		num_ims, rows, cols, _ = X.shape
		
		predictions = np.zeros_like(Y)
		prediction_probabilities = np.zeros((Y.shape[0], Y.shape[1], Y.shape[2], self.pt.class_n), dtype = np.float16)
		for im_id in range(num_ims):
			x = X[im_id][np.newaxis,...]
			# ic(x.shape)
			prediction_probability = self.model.predict(x)
			prediction = prediction_probability.argmax(axis=-1)
			predictions[im_id] = prediction.copy()
			if save_prediction_probability == True:
				prediction_probabilities[im_id] = prediction_probability.copy()

		if save_prediction_probability == True:		
			return predictions, prediction_probabilities
		return predictions

	def loadIntermediateFeatures(self, in_, prediction_dtype = np.float16, debug  = 1):
	#print(model.summary())

#		layer_names = ['conv_lst_m2d_1', 'activation_6', 'activation_8', 'activation_10']
		# layer_names = ['conv_lst_m2d', 'activation_5', 'activation_7', 'activation_9']
		layer_names = ['conv2d_7', 'conv2d_10', 'conv2d_13', 'conv2d_16']
		upsample_ratios = [8, 4, 2, 1]

		out1 = UpSampling2D(size=(upsample_ratios[0], upsample_ratios[0]))(self.model.get_layer(layer_names[0]).output)
		out2 = UpSampling2D(size=(upsample_ratios[1], upsample_ratios[1]))(self.model.get_layer(layer_names[1]).output)
		out3 = UpSampling2D(size=(upsample_ratios[2], upsample_ratios[2]))(self.model.get_layer(layer_names[2]).output)
		out4 = UpSampling2D(size=(upsample_ratios[3], upsample_ratios[3]))(self.model.get_layer(layer_names[3]).output)

		intermediate_layer_model = Model(inputs=self.model.input, outputs=[out1, #4x4
															out2, #8x8
															out3, #16x16
															out4]) #32x32

		intermediate_features=intermediate_layer_model.predict(in_) 

		if debug > 0:
			deb.prints(intermediate_features[0].shape)

		intermediate_features = [x.reshape(x.shape[0], -1, x.shape[-1]) for x in intermediate_features]
		if debug > 0:
			[deb.prints(intermediate_features[x].shape) for x in [0,1,2,3]]

		intermediate_features = np.squeeze(np.concatenate(intermediate_features, axis=-1))# .astype(prediction_dtype)


		if debug > 0:
			deb.prints(intermediate_features.shape)
			print("intermediate_features stats", np.min(intermediate_features), np.average(intermediate_features), np.max(intermediate_features))
		return intermediate_features
