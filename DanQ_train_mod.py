import numpy as np
import h5py
import sys
import pickle
import os.path
import scipy.io
np.random.seed(1337) # for reproducibility

if len(sys.argv)==2:
    print 'training iteration number is'+sys.argv[1]+'\n' 
    train_num=int(sys.argv[1])
else:
    print 'No command line arguments provided'
    print 'Length of sys.argv: '+str(len(sys.argv))
    print sys.argv
    train_num=1

    
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



print 'loading data'
trainmat = h5py.File('data/deepsea_train/train.mat')
validmat = scipy.io.loadmat('data/deepsea_train/valid.mat')
testmat = scipy.io.loadmat('data/deepsea_train/test.mat')

X_train = np.transpose(np.array(trainmat['trainxdata']),axes=(2,0,1))
y_train = np.array(trainmat['traindata']).T

# forward_lstm = LSTM(input_dim=320, output_dim=320, return_sequences=True)
# backward_lstm = LSTM(input_dim=320, output_dim=320, return_sequences=True)
# brnn = Bidirectional(forward=forward_lstm, backward=backward_lstm, return_sequences=True)

print 'building model'
from keras.models import model_from_json

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

model.add(Bidirectional(LSTM(input_dim=320,output_dim=320,return_sequences=True,init='he_normal')))

model.add(Dropout(0.01))

model.add(Flatten())

model.add(Dense(input_dim=75*640, output_dim=925,init='he_normal'))
model.add(LeakyReLU(alpha=0.01))

model.add(Dense(input_dim=925, output_dim=919,init='he_normal'))
model.add(Activation('sigmoid'))

print 'compiling model'
model.compile(loss='binary_crossentropy', optimizer='rmsprop')

print 'checking for pre-existing models'
prev_model = "DanQ_bestmodel"+str(train_num-1)+".hdf5"
if os.path.exists(prev_model):
    print 'found previous model: '+prev_model+'\n'
    model.load_weights(prev_model)
else:
    print 'no previous model found for: '+prev_model+'\n'

print 'running 1 epoch'
model_hist = "DanQ_modelhist"+str(train_num)+"_mod.p"
checkpointer = ModelCheckpoint(filepath="DanQ_bestmodel"+str(train_num)+"_mod.hdf5", verbose=1, save_best_only=True)
earlystopper = EarlyStopping(monitor='val_loss', patience=5)

history = model.fit(X_train, y_train, batch_size=200, nb_epoch=1, shuffle=True, validation_data=(np.transpose(validmat['validxdata'],axes=(0,2,1)), validmat['validdata']), callbacks=[checkpointer,earlystopper])
print 'pickling epoch history'
pickle.dump(history.history,open(model_hist,"wb"))


tresults = model.evaluate(np.transpose(testmat['testxdata'],axes=(0,2,1)), testmat['testdata'])

print tresults

