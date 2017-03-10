import numpy as np
import tables
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
from keras.callbacks import ModelCheckpoint, EarlyStopping
from seya.layers.recurrent import Bidirectional
#import theano
from keras.utils.layer_utils import print_layer_shapes



print 'loading data'
trainmat = h5py.File('Noonan_train.h5')
X_train = np.array(trainmat['trainxdata'])
y_train = np.array(trainmat['traindata'])
print 'X_train shape:'+str(X_train.shape)
print 'y_train shape:'+str(y_train.shape)


validmat = h5py.File('Noonan_valid.h5')
X_valid= np.array(validmat['validxdata'])
y_valid=np.array(validmat['validdata'])


testmat = h5py.File('Noonan_test.h5')
X_test= np.array(testmat['testxdata'])
y_test=np.array(testmat['testdata'])

print 'Building model'

forward_lstm = LSTM(input_dim=320, output_dim=320, return_sequences=True)
forward_lstm.trainable=False
backward_lstm = LSTM(input_dim=320, output_dim=320, return_sequences=True)
backward_lstm.trainable=False
brnn = Bidirectional(forward=forward_lstm, backward=backward_lstm, return_sequences=True)
brnn.trainable=False

print 'building model'


model = Sequential()
model.add(Convolution1D(input_dim=4,
                        input_length=1000,
                        nb_filter=320,
                        filter_length=26,
                        border_mode="valid",
                        activation="relu",
                        subsample_length=1,trainable=False))

model.add(MaxPooling1D(pool_length=13, stride=13,trainable=False))

model.add(Dropout(0.2))


model.add(brnn)

model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(input_dim=75*640, output_dim=925,trainable=False))
model.add(Activation('relu'))

model.add(Dense(input_dim=925, output_dim=919))
model.add(Activation('sigmoid'))
print 'loading weights'
model.load_weights('DanQ_bestmodel.hdf5')
print 'popping last layer'
model.layers.pop()
print 'adding back final layer'
model.add(Dense(input_dim=925, output_dim=1))
model.add(Activation('sigmoid'))

print 'compiling model'
model.compile(loss='binary_crossentropy', optimizer='rmsprop', class_mode="binary")





print 'running at most 5 epochs'

checkpointer = ModelCheckpoint(filepath="DanQ_Noonan_bestmodel.hdf5", verbose=1, save_best_only=True)
earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

model_hist = model.fit(X_train, y_train, batch_size=100, nb_epoch=5, shuffle=True, show_accuracy=True, validation_data=(X_valid,y_valid), callbacks=[checkpointer,earlystopper])

model

tresults = model.evaluate(X_test, y_test,show_accuracy=True)

print tresults
