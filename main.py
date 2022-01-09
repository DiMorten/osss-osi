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
#ic(tf.config.list_physical_devices('GPU'))
#pdb.set_trace()
paramsTrainCustom = {
    "dataPath": Path("D:/jorg/phd/dataset_original/")
}

pt = paramsTrain(**paramsTrainCustom)

ds = Dataset(pt)
ds.load()

ds.trainValSplit(0.15)

ds.extractCoords()
 # to do: extracts train and also test coords

ds.addPadding()


ds.setTrainGenerator(rows = pt.h,
        cols = pt.w,
        num_ims = pt.num_ims_train,
        patch_size = pt.patch_size)

modelManager = ModelManager(pt)
modelManager.setArchitecture(Unet)

modelManager.configure()
modelManager.fit(ds.trainGenerator, ds.validationGenerator)
pdb.set_trace()
'''
arch = Unet()

modelManager = ModelManager()
modelManager.setArch(arch)
modelManager.setData(ds)
modelManager.configure()
modelManager.fit()
'''