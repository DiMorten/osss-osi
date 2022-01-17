
import numpy as np
from time import time
import numpy as np
import tensorflow.keras.backend as K
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv3D, Conv3DTranspose, AveragePooling3D
from tensorflow.keras.layers import AveragePooling2D, Flatten, BatchNormalization, Dropout, TimeDistributed, ConvLSTM2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import ELU, Lambda
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, jaccard_score
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import f1_score, make_scorer, confusion_matrix, accuracy_score, precision_score, recall_score, precision_recall_curve
import pdb
from tensorflow.keras.regularizers import l1,l2
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Dropout, Conv2DTranspose, AveragePooling2D, Bidirectional, Activation
from icecream import ic
from pathlib import Path
import cv2
import joblib

from icecream import ic


class Monitor(Callback):
    def __init__(self, validation, classes):
        super(Monitor, self).__init__()
        self.validation = validation 
        self.classes = classes

    def getValidationData(self):
        for batch_index in range(len(self.validation)):
            val_targ = self.validation[batch_index][1]   
            val_pred = self.model.predict(self.validation[batch_index][0])
            val_prob = val_pred.copy()
            val_predict = np.argmax(val_prob,axis=-1)

            val_targ = np.squeeze(val_targ)
            #ic(val_predict.shape, val_targ.shape)
            val_predict = val_predict[val_targ<self.classes]
            val_targ = val_targ[val_targ<self.classes]
            self.pred.extend(val_predict)
            self.targ.extend(val_targ)       

    def on_epoch_begin(self, epoch, logs={}):        
        self.pred = []
        self.targ = []

    def on_epoch_end(self, epoch, logs={}):
        self.getValidationData()
        f1 = np.round(f1_score(self.targ, self.pred, average=None)*100,2)
        ic(f1)