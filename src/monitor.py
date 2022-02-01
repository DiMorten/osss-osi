
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
import matplotlib.pyplot as plt
from pathlib import Path

def plot_sample_ims(in_, targ, pred, epoch, batch_id):
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(9, 6))
    # ic(len(targ), len(pred))
    # pdb.set_trace()
    targ = np.squeeze(np.asarray(targ))
    pred = np.asarray(pred).argmax(axis=-1)
    in_ = np.asarray(in_).astype(np.float32)
    
    max_id = targ.shape[0]
    isAutoSnipIdMode = True # [4, 13, 14]
    if isAutoSnipIdMode == True:
        snip_id = None
        for id_ in range(max_id):
            if np.any(targ[id_] != 0):
                # snip_ids.append(id_)
                snip_id = id_
                break
        if snip_id == None: snip_id = 0

        # ic(snip_ids)
        # pdb.set_trace()
        # snip_id = snip_ids[0]
    else:
        snip_id = 4
    ic(snip_id)

    targ, pred, in_ = targ[snip_id], pred[snip_id], in_[snip_id]
    # cmaps = ['gray', ]
    # ic(in_.shape, targ.shape, pred.shape)
    # ic(in_.dtype, targ.dtype, pred.dtype)

    targ = targ.astype(np.float32)

    # pred = np.asarray(pred).astype(np.float32)
    # ic(pred.shape, pred.dtype)
#     pdb.set_trace()
#    .argmax(axis=-1).astype(np.float32)
    display_ims = [in_, targ, pred]
    
    for idx, ax in enumerate(axes.flat):
        ax.set_axis_off()
        # ic(display_ims[idx].shape)
        # ic(display_ims[idx].dtype)
        im = ax.imshow(display_ims[idx]) # , cmap = 'gray'
        if idx == 2: break
    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                            wspace=0.02, hspace=0.02)
    
    plt.axis('off')
    results_dir = Path('results/validation_snips')
    # plt.show()
    plt.savefig(results_dir / 
        ('batch_id' + str(batch_id) + '_epoch' + str(epoch) + '.png'), 
        dpi = 300, format='png', bbox_inches = 'tight')
    plt.clf()
    plt.close()
    #pdb.set_trace()

class Monitor(Callback):
    def __init__(self, validation, classes):
        super(Monitor, self).__init__()
        self.validation = validation 
        self.classes = classes

    def getValidationData(self):
        # snip_batch_id = np.random.choice(len(self.validation))
        #snip_batch_ids = [0, 5, 10]
        snip_batch_ids = [0, 1, 2]
        ic(len(self.validation))
        for batch_index in range(len(self.validation)):
            val_targ = self.validation[batch_index][1] 
            val_in = self.validation[batch_index][0]  
            val_pred = self.model.predict(val_in)
            val_prob = val_pred.copy()
            val_predict = np.argmax(val_prob,axis=-1)
            if batch_index in snip_batch_ids:
                plot_sample_ims(val_in, val_targ, val_pred, self.epoch,
                    batch_index)
            val_targ = np.squeeze(val_targ)
            #ic(val_predict.shape, val_targ.shape)

            self.pred.extend(val_predict)
            self.targ.extend(val_targ)       

    def on_epoch_begin(self, epoch, logs={}):        
        self.pred = []
        self.targ = []
        self.epoch = epoch

    def on_epoch_end(self, epoch, logs={}):
        self.getValidationData()
        ic(len(self.targ), len(self.pred))
        # plot_sample_ims(self.targ, self.pred)
        self.targ = np.asarray(self.targ)
        self.pred = np.asarray(self.pred)
        ic(self.targ.shape, self.pred.shape)
        ic(np.unique(self.pred, return_counts=True),
            np.unique(self.targ, return_counts=True))        
        self.pred = self.pred[self.targ<self.classes]
        self.targ = self.targ[self.targ<self.classes]
        ic(self.targ.shape, self.pred.shape)
        # ic(np.unique(self.pred, return_counts=True),
        #     np.unique(self.targ, return_counts=True))

        f1 = np.round(f1_score(self.targ, self.pred, average=None)*100,2)
        ic(f1)