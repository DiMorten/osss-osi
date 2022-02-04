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

def resnet_block(x, n_filter, ind):
    x_init = x

    ## Conv 1
    x = Conv2D(n_filter, (3, 3), activation='relu', padding="same", name = 'res1_net'+str(ind))(x)
    x = Dropout(0.5, name = 'drop_net'+str(ind))(x, training = True)
    # x = SpatialDropout2D(0.5, name = 'drop_net'+str(ind))(x, training = True)

    ## Conv 2
    x = Conv2D(n_filter, (3, 3), activation='relu', padding="same", name = 'res2_net'+str(ind))(x)
    
    ## Shortcut
    s  = Conv2D(n_filter, (1, 1), activation='relu', padding="same", name = 'res3_net'+str(ind))(x_init)
    
    ## Add
    x = Add()([x, s])
    return x
    
class ModelArchitecture():
     def __init__(self, pt):
        self.pt = pt


class ResUnet():
    def __init__(self, img_shape = (128,128,25),class_n=10):
        self.img_shape = img_shape
        self.class_n = class_n

    def build(self, nb_filters = [16, 32, 64, 128, 256]):
        nb_filters = [64, 128, 256, 128, 64]
        '''Base network to be shared (eq. to feature extraction)'''
        #nb_filters = [16, 32, 64, 128]
        input_img = Input(shape = self.img_shape, name="input_enc_net")
        # ic(K.int_shape(input_img))
        res_block1 = resnet_block(input_img, nb_filters[0], 1) 
        pool1 = MaxPool2D((2 , 2), name='pool_net1')(res_block1)
        # ic(K.int_shape(pool1))
        res_block2 = resnet_block(pool1, nb_filters[1], 2)
        pool2 = MaxPool2D((2 , 2), name='pool_net2')(res_block2)
        # ic(K.int_shape(pool2))
        res_block3 = resnet_block(pool2, nb_filters[2], 3)
        pool3 = MaxPool2D((2 , 2), name='pool_net3')(res_block3)
        # ic(K.int_shape(pool3))
        res_block4 = resnet_block(pool3, nb_filters[3], 4)
        pool4 = MaxPool2D((2 , 2), name='pool_net4')(res_block4)
        
        res_block5 = resnet_block(pool4, nb_filters[4], 5)
        # ic(K.int_shape(res_block5))
        #res_block6 = resnet_block(res_block5, nb_filters[2], 6)
        
        upsample4 = Conv2D(nb_filters[3], (3 , 3), activation = 'relu', padding = 'same', 
                        name = 'upsampling_net4')(UpSampling2D(size = (2,2))(res_block5))
        # ic(K.int_shape(upsample4))
        merged4 = concatenate([res_block4, upsample4], name='concatenate4')
        
        upsample3 = Conv2D(nb_filters[2], (3 , 3), activation = 'relu', padding = 'same', 
                        name = 'upsampling_net3')(UpSampling2D(size = (2,2))(merged4))
        
        merged3 = concatenate([res_block3, upsample3], name='concatenate3')

        upsample2 = Conv2D(nb_filters[1], (3 , 3), activation = 'relu', padding = 'same', 
                        name = 'upsampling_net2')(UpSampling2D(size = (2,2))(merged3))
                                                    
        merged2 = concatenate([res_block2, upsample2], name='concatenate2')
                                                                                            
        upsample1 = Conv2D(nb_filters[0], (3 , 3), activation = 'relu', padding = 'same', 
                        name = 'upsampling_net1')(UpSampling2D(size = (2,2))(merged2))
        merged1 = concatenate([res_block1, upsample1], name='concatenate1')

        output = Conv2D(self.class_n,(1,1), activation = 'softmax', padding = 'same', name = 'output')(merged1)
                                                                                                            
        model = Model(input_img, output)
        ic(model.summary())
        return model

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
