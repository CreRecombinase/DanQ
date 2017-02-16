import sys
import numpy as np
import h5py
import scipy.io
np.random.seed(1337) # for reproducibility




from keras.models import load_model

from keras.preprocessing import sequence
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Bidirectional
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.regularizers import l2, activity_l1
from keras.constraints import maxnorm
from keras.layers.recurrent import LSTM, GRU
from keras.layers.advanced_activations import LeakyReLU, PReLU

if len(sys.argv)==2:
    print 'training iteration number is'+sys.argv[1]+'\n' 
    train_num=int(sys.argv[1])
else:
    print 'No command line arguments provided'
    print 'Length of sys.argv: '+str(len(sys.argv))
    print sys.argv
    train_num=1




#bmodelf = 'DanQ_bestmodel1_mod.hdf5'
#model = load_model(bmodelf)



print 'building model'

model = Sequential()
model.add(Convolution1D(input_dim=4,
                        input_length=1000,
                        nb_filter=320,
                        filter_length=26,
                        border_mode="valid",
                        subsample_length=1,init='he_normal'))
model.add(LeakyReLU(alpha=0.01))

model.add(MaxPooling1D(pool_length=13, stride=13))

model.add(Dropout(0.01))

model.add(Bidirectional(LSTM(input_dim=320,output_dim=320,return_sequences=True,init='he_normal')))

model.add(Dropout(0.01))

model.add(Flatten())

model.add(Dense(input_dim=75*640, output_dim=925,init='he_normal'))
model.add(LeakyReLU(alpha=0.01))

model.add(Dense(input_dim=925, output_dim=919,init='he_normal'))
model.add(Activation('sigmoid'))

print 'compiling model'
model.compile(loss='binary_crossentropy', optimizer='rmsprop')



model.load_weights('DanQ_bestmodel1_mod.hdf5')

print 'loading test data'
testmat = h5py.File(testf,'r')
x = np.transpose(testmat['testxdata'].value,axes=(0,2,1))
testmat.close()



print 'loading test data'
testmat = scipy.io.loadmat('data/deepsea_train/test.mat')

print 'evalating on test data'
tresults = model.evaluate(np.transpose(testmat['testxdata'],axes=(0,2,1)), testmat['testdata'])
print tresults



print 'predicting on test sequences'
y = model.predict(x, verbose=1)

print "saving to " + sys.argv[2]
f = h5py.File(sys.argv[2], "w")
f.create_dataset("pred", data=y)
f.close()
