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

from src.generator import DataGeneratorWithCoords
from sklearn.model_selection import train_test_split
def extract_coords(num_ims, rows, cols, patch_size, overlap_percent=0):

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
                patch_coords.append((im_id, i*stride, j*stride))
    print("Patch coords ",len(patch_coords))
    return np.asarray(patch_coords), step_row, step_col, overlap, pad_tuple_mask

class Dataset():
    def __init__(self, pt):
        self.pt = pt
    def load(self):
        ic(self.pt.dataPath / "train")
        x_list = glob.glob(str(self.pt.dataPath / "train" / "images") + "/*")
        y_list = glob.glob(str(self.pt.dataPath / "train" / "labels_1D") + "/*")
        X = []
        Y = []
        for idx, (x_name, y_name) in enumerate(zip(x_list, y_list)):
            x = cv2.imread(x_name)
            y = cv2.imread(y_name)
            #ic(x.shape, y.shape)
            X.append(x)
            Y.append(y)
#            pdb.set_trace()
        self.X_train = np.stack(X, axis = 0)
        self.Y_train = np.stack(Y, axis = 0)
        ic(self.X_train.shape, self.Y_train.shape)
        self.Y_train = self.Y_train[...,0]



    def extractCoords(self, num_ims = 1002, rows = 650, cols = 1250, patch_size = 250, overlap_percent=0):


        """
        Everything  in this function is made operating with
        the upper left corner of the patch
        """

        self.patch_coords_train, step_row, step_col, overlap, self.pad_tuple_mask = extract_coords(
            self.X_train.shape[0], self.pt.h, self.pt.w, self.pt.patch_size, overlap_percent=0)

        ic(self.patch_coords_train.shape, step_row, step_col, overlap, self.pad_tuple_mask)

        self.patch_coords_validation, step_row, step_col, overlap, self.pad_tuple_mask = extract_coords(
            self.X_validation.shape[0], self.pt.h, self.pt.w, self.pt.patch_size, overlap_percent=0)

        ic(self.patch_coords_validation.shape, step_row, step_col, overlap, self.pad_tuple_mask)
  
    def addPadding(self):
        pad_tuple = ((0, 0),) + self.pad_tuple_mask + ((0, 0),) 
        Y_pad_tuple = ((0, 0),) + self.pad_tuple_mask

        ic(pad_tuple)
        self.X_train = np.pad(self.X_train, pad_tuple, mode = 'symmetric')
        ic(self.Y_train.shape)
        self.Y_train = np.pad(self.Y_train, Y_pad_tuple, mode = 'symmetric')

        self.X_validation = np.pad(self.X_validation, pad_tuple, mode = 'symmetric')
        ic(self.Y_validation.shape)
        self.Y_validation = np.pad(self.Y_validation, Y_pad_tuple, mode = 'symmetric')


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
			'n_channels': 3,
			'shuffle': True,
			'augm': True,
            'subsample': False}        
        
        ic(self.X_train.shape, self.Y_train.shape)
        ic(np.unique(self.Y_train, return_counts=True))

        self.trainGenerator = DataGeneratorWithCoords(self.X_train, self.Y_train, 
					self.patch_coords_train, **params_train)

        params_validation = params_train.copy()
        params_validation['shuffle'] = False
        params_validation['augm'] = False
        params_validation['subsample'] = False

        self.validationGenerator = DataGeneratorWithCoords(self.X_validation, self.Y_validation, 
					self.patch_coords_validation, **params_validation)

        params_test = params_validation.copy()
        self.testGenerator = DataGeneratorWithCoords(self.X_test, self.Y_test, 
					self.patch_coords_test, **params_test)


    def trainValSplit(self, validation_split=0.15):
        idxs = range(self.pt.num_ims_train)
        self.X_train, self.X_validation, self.Y_train, self.Y_validation = train_test_split(
                self.X_train, self.Y_train, test_size = validation_split)
        self.num_ims_train = self.X_train.shape[0]
        ic(self.X_train.shape, self.X_validation.shape, self.Y_train.shape, self.Y_validation.shape)
        
