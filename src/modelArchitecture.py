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

class ModelArchitecture():
     def __init__(self, pt):
        self.pt = pt

class Unet():
    def __init__(self, img_shape = (128,128,25),class_n=10):
        self.img_shape = img_shape
        self.class_n = class_n

    def build(self, pretrained_weights = None):
        inputs = Input(shape=self.img_shape)
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        drop3 = Dropout(0.5)(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)

        conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)
        
        # Classification branch
        up6 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
        merge6 = concatenate([drop3,up6], axis = 3)
        conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

        up7 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        merge7 = concatenate([conv2,up7], axis = 3)
        conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

        up8 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        merge8 = concatenate([conv1,up8], axis = 3)
        conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
        classfier = Conv2D(self.class_n, 1, activation = 'softmax', name='cl_output')(conv8)
        
        
        # Regression branch
    #    up61 = Conv2D(24, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    #    merge61 = concatenate([drop3,up61], axis = 3)
    #    conv61 = Conv2D(24, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge61)
    #    conv61 = Conv2D(24, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv61)

    #    up71 = Conv2D(24, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv61))
    #    merge71 = concatenate([conv2,up71], axis = 3)
    #    conv71 = Conv2D(24, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge71)
    #    conv71 = Conv2D(24, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv71)

    #    up81 = Conv2D(24, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv71))
    #    merge81 = concatenate([conv1,up81], axis = 3)
    #    conv81 = Conv2D(24, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge81)
    #    conv81 = Conv2D(24, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv81)
    #    regresor = Conv2D(1, 1, activation = 'sigmoid', name='reg_output')(conv81)


        model = Model(inputs = inputs, outputs = [classfier])
        print(model.summary())
        return model
