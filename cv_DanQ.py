from __future__ import print_function
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
import pickle



def data():
    '''
    Data providing function:

    Make sure to have every relevant import statement included here and return data as
    used in model function below. This function is separated from model() so that hyperopt
    won't reload data for each evaluation run.
    '''
    
    import numpy as np
    import tables
    import h5py
    import scipy.io
    np.random.seed(1337)
    trainmat = h5py.File('data/deepsea_train/train.mat')
    testmat = scipy.io.loadmat('data/deepsea_train/test.mat')
    
    full_X_train = np.transpose(np.array(trainmat['trainxdata']),axes=(2,0,1))
    sub_sample=np.random.choice(full_X_train.shape[0],733333)
    X_train = full_X_train[sub_sample,:,:]

    full_y_train = np.array(trainmat['traindata']).T
    Y_train = full_y_train[sub_sample,:]
    
    X_test= np.transpose(testmat['testxdata'],axes=(0,2,1))
    Y_test= testmat['testdata']
    trainmat.close()
    
    return X_train, Y_train, X_test, Y_test


def model(X_train, Y_train, X_test, Y_test):
    '''
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    '''
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation
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

    model = Sequential()

    
    model = Sequential()
    model.add(Convolution1D(input_dim=4,
                            input_length=1000,
                            nb_filter=320,
                            filter_length=26,
                            border_mode="valid",
                            subsample_length=1,init='he_normal'))
    if conditional({{choice(['relu','leakyrelu'])}}) =='leakyrelu':
       model.add(LeakyReLU(alpha={{uniform(0,1)}}))
    else:
       model.add(Activation('relu'))
       

    model.add(MaxPooling1D(pool_length=13, stride=13))

    model.add(Dropout({{uniform(0,1)}}))
    model.add(Bidirectional(LSTM(input_dim=320,output_dim=320,
                                 return_sequences=True,init='he_normal')))

    model.add(Dropout({{uniform(0,1)}}))

    model.add(Flatten())

    if conditional({{choice(['dense','double'])}}) == 'double':
       model.add(Dense(input_dim=75*640, output_dim=925,init='he_normal'))
       model.add(LeakyReLU(alpha={{uniform(0,1)}}))

       model.add(Dense(input_dim=925, output_dim=919,init='he_normal'))
       model.add(Activation('tanh'))
    else:
       model.add(Dense(input_dim=75*640, output_dim=919,init='he_normal'))
       model.add(Activation('tanh'))


    model.compile(loss='binary_crossentropy', optimizer={{choice(['rmsprop', 'adam', 'sgd'])}},metrics=['accuracy'])

    model.fit(X_train, Y_train,
              batch_size={{choice([64, 128])}},
              nb_epoch=1,
              shuffle=True,
              validation_data=(X_test, Y_test))
    score, acc = model.evaluate(X_test, Y_test)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())
    pickle.dump(best_run,open("cv_DanQ_best_run.p","wb"))
    pickle.dump(best_model,open("cv_DanQ_best_model.p","wb"))
    X_train, Y_train, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
