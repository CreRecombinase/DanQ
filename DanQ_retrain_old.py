import numpy as np
import h5py
import sys
import pickle
import os.path
import scipy.io
np.random.seed(1337) # for reproducibility

  
from keras.preprocessing import sequence
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Bidirectional
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.regularizers import l2, activity_l1
from keras.constraints import maxnorm
from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.models import model_from_json


print 'building model'

with open("pre_brnn.json",'r') as tfile:
    fmodelj=tfile.read()
    
fmodel = from_json(fmodelj)

model = Sequential()
model.add(Convolution1D(input_dim=4,
                        input_length=1000,
                        nb_filter=320,
                        filter_length=26,
                        border_mode="valid",
                        activation="relu",
                        subsample_length=1,init='he_normal'))
#model.add(LeakyReLU(alpha=0.01))

model.add(MaxPooling1D(pool_length=13, stride=13))

model.add(Dropout(0.2))
model.load_weights('pre_brnn_weights.h5')
