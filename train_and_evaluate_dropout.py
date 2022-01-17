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
from pathlib import Path
import pdb
from src.modelArchitecture import Unet
tf.random.set_seed(2)
np.random.seed(2)

t0 = time.time()

import os
os.getcwd()

from src.dataset import Dataset, extract_coords
from params.paramsTrain import paramsTrain
from src.modelManager import ModelManager
from train_and_evaluate import Manager
#ic(tf.config.list_physical_devices('GPU'))

ic.configureOutput(includeContext=True)

class ManagerDropout(Manager):
	def evaluate(self, ds):
		ds = self.modelManager.evaluateEnsemble(ds)
		np.save('results/predictions_ensemble.npy', ds.predictions_ensemble)
		pdb.set_trace()
		

if __name__ == '__main__':
	paramsTrainCustom = {
	"dataPath": Path("D:/jorg/phd/dataset_original/"),
	# "loss": "weighted_categorical_crossentropy",
	"loss": "categorical_focal",

	"mode": "inference",  # train, inference
	"modelId": ""
	}

	pt = paramsTrain(**paramsTrainCustom)

	manager = ManagerDropout(pt)

	manager.main()




