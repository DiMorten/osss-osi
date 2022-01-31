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
from src.modelArchitecture import Unet, ResUnet
tf.random.set_seed(2)
np.random.seed(2)

t0 = time.time()

import os
os.getcwd()

from src.dataset import Dataset, extract_coords
from params.paramsTrain import paramsTrain
from src.modelManager import ModelManager
#ic(tf.config.list_physical_devices('GPU'))

ic.configureOutput(includeContext=True)

class Manager():
        def __init__(self, pt):
                self.pt = pt
        def evaluate(self, ds):
                self.modelManager.evaluate(ds)
        def main(self):
                ds = Dataset(self.pt)
                ds.load()
                ds.normalize()

                ds.trainValSplit(0.15)

                ds.extractCoords()

                ds.addPadding()
                ds.trainReduce(trainSize = 64)
                self.pt.num_ims_train = 64
                ds.useLabelsAsInput()

                ds.setTrainGenerator(rows = self.pt.h,
                        cols = self.pt.w,
                        num_ims = self.pt.num_ims_train,
                        patch_size = self.pt.patch_size)

                self.modelManager = ModelManager(self.pt)
                self.modelManager.setArchitecture(ResUnet)
                ic(ds.Y_train.shape)
                ic(np.unique(ds.Y_train, return_counts=True))
                ic(np.unique(ds.Y_validation, return_counts=True))
                ic(np.unique(ds.Y_test, return_counts=True))
                 
                self.modelManager.computeWeights(ds.Y_train.flatten())
                
                self.modelManager.configure()
                if self.pt.mode == "train":
                        
                        # self.modelManager.fit(ds.trainGenerator, ds.validationGenerator)
                        self.modelManager.fit(ds.trainGenerator, ds.trainGenerator)
                
                self.modelManager.loadWeights()

                self.evaluate(ds)        
                        
                pdb.set_trace()

if __name__ == '__main__':
        paramsTrainCustom = {
        "dataPath": Path("D:/jorg/phd/dataset_original/"),
        #"loss": "categorical_focal", # available: "categorical_crossentropy", "categorical_focal", "weighted_categorical_crossentropy"     
        "loss": "weighted_categorical_crossentropy", # available: "categorical_crossentropy", "categorical_focal", "weighted_categorical_crossentropy"             
        "mode": "train",  # mode: train, inference
        "modelId": "weighted"
        }

        pt = paramsTrain(**paramsTrainCustom)

        manager = Manager(pt)

        manager.main()