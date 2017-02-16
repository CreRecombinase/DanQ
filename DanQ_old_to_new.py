import sys
import numpy as np
import h5py
import scipy.io
np.random.seed(1337) # for reproducibility

from keras.preprocessing import sequence
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.regularizers import l2, activity_l1
from keras.constraints import maxnorm
from keras.layers.recurrent import LSTM, GRU
from seya.layers.recurrent import Bidirectional

import theano

forward_lstm = LSTM(input_dim=320, output_dim=320, return_sequences=True)
backward_lstm = LSTM(input_dim=320, output_dim=320, return_sequences=True)
brnn = Bidirectional(forward=forward_lstm, backward=backward_lstm, return_sequences=True)

print 'building model'

model = Sequential()
model.add(Convolution1D(input_dim=4,
                        input_length=1000,
                        nb_filter=320,
                        filter_length=26,
                        border_mode="valid",
                        activation="relu",
                        subsample_length=1))

model.add(MaxPooling1D(pool_length=13, stride=13))

model.add(Dropout(0.2))

model.add(brnn)

model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(input_dim=75*640, output_dim=925))
model.add(Activation('relu'))

model.add(Dense(input_dim=925, output_dim=919))
model.add(Activation('sigmoid'))

print 'compiling model'
#model.compile(loss='binary_crossentropy', optimizer='rmsprop', class_mode="binary")


model.load_weights('DanQ_bestmodel.hdf5')

fmodel = Sequential(model.layers[0:3])
fmodelj = fmodel.to_json()
fmodel.save_weights('pre_brnn_weights.h5')
with open("pre_brnn.json",'w') as tfile:
    tfile.write(fmodelj)
    


rmodel = Sequential(model.layers[4:])
rmodel.save_weights('post_brnn_weights.h5')
rmpdelj = rmodel.to_json()
with open("post_brnn.json",'w') as tfile:
    tfile.write(rmodelj)
                    
                    
