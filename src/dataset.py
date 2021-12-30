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
def extractCoords(num_ims, rows, cols, patch_size, overlap_percent=0):

    """
    Everything  in this function is made operating with
    the upper left corner of the patch
    """


    # Percent of overlap between consecutive patches.
    overlap = round(patch_size * percent)
    overlap -= overlap % 2
    stride = patch_size - overlap
    # Add Padding to the image to match with the patch size
    step_row = (stride - rows % stride) % stride
    step_col = (stride - cols % stride) % stride
    pad_tuple_msk = ( (overlap//2, overlap//2 + step_row), ((overlap//2, overlap//2 + step_col)) )
#    lbl = np.pad(lbl, pad_tuple_msk, mode = 'symmetric')

    k1, k2 = (rows+step_row)//stride, (cols+step_col)//stride
    print('Total number of patches: %d x %d' %(k1, k2))

    patch_coords = []
    for im_id in range(num_ims):
        for i in range(k1):
            for j in range(k2):
                patch_coords.append((im_id, i*stride, j*stride))
    print("Patch coords ",len(patch_coords))
    return patch_coords, step_row, step_col, overlap

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
        self.X = np.stack(X, axis = 0)
        self.Y = np.stack(Y, axis = 0)
        ic(self.X.shape, self.Y.shape)


    def extractCoords(self, num_ims = 1002, rows = 650, cols = 1250, patch_size = 250, overlap_percent=0):


        """
        Everything  in this function is made operating with
        the upper left corner of the patch
        """


        # Percent of overlap between consecutive patches.
        self.overlap = round(patch_size * overlap_percent)
        self.overlap -= self.overlap % 2
        stride = patch_size - self.overlap
        # Add Padding to the image to match with the patch size
        self.step_row = (stride - rows % stride) % stride
        self.step_col = (stride - cols % stride) % stride
        self.pad_tuple_msk = ( (self.overlap//2, self.overlap//2 + self.step_row), (self.overlap//2, self.overlap//2 + self.step_col) )
    #    lbl = np.pad(lbl, pad_tuple_msk, mode = 'symmetric')

        k1, k2 = (rows+self.step_row)//stride, (cols+self.step_col)//stride
        print('Total number of patches: %d x %d' %(k1, k2))

        self.patch_coords = []
        for im_id in range(num_ims):
            for i in range(k1):
                for j in range(k2):
                    self.patch_coords.append((im_id, i*stride, j*stride))
        self.patch_coords = np.asarray(self.patch_coords)
        ic(self.patch_coords.shape, self.step_row, self.step_col, self.overlap)
    
    def addPadding(self):
        pad_tuple = ((0, 0),) + self.pad_tuple_msk + ((0, 0),) 
        ic(pad_tuple)
        self.X = np.pad(self.X, pad_tuple, mode = 'symmetric')
        self.Y = np.pad(self.Y, pad_tuple, mode = 'symmetric')
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
#			'n_classes': self.class_n,
			'n_classes': self.pt.class_n, # it was 6. Now it is 13 + 1 = 14

			'n_channels': 3,
			'shuffle': True,
#			'printCoords': False,
			'augm': True}        
        self.Y = self.Y[...,0]
        ic(self.X.shape, self.Y.shape)
        ic(np.unique(self.Y, return_counts=True))
        '''
        im_n, h, w = self.Y.shape
#        unique = np.unique(self.Y)
#        class_n = len(unique)        
        self.Y = np.reshape(self.Y, -1)
        self.Y = self.toOneHot(self.Y, class_n = self.pt.class_n)
        self.Y = np.reshape(self.Y, (im_n, h, w, self.pt.class_n))
        ic(self.X.shape, self.Y.shape)
        '''
        self.trainGenerator = DataGeneratorWithCoords(self.X, self.Y, 
					self.patch_coords, **params_train)

