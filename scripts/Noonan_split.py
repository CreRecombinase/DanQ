import numpy as np
import tables
import h5py
import scipy.io
np.random.seed(1337) # for reproducibility
from sklearn.model_selection import train_test_split




def write_mat_h5(h5filename,groupname,dataname,data,axis=None):
    with h5py.File(h5filename,mode='a') as tf:
        print(dataname)
        grppth='/'+groupname
        tpth = grppth+'/'+dataname
        drows,dcols=data.shape
        if not grppth in tf:
            grp = tf.create_group(groupname)
        else:
            grp=tf[grppth]
        if data.dtype=='O':
            data=data.astype("str")
        if not tpth in tf:
                dset = grp.create_dataset(tpth,data.shape,
                                          chunks=True,
                                          maxshape=(None,None),
                                          compression=32001,
                                          compression_opts=(0, 0, 0, 0, 3, 2, 0),
                                          shuffle=True,
                                          data=data)
        else:
            dset=grp[dataname]
            print(dset)
            orows,ocols=dset.shape
            if axis is None:
                nrows=orows+drows
                ncols=ocols+dcols
                dset.resize((nrows,ncols))
                dset[orows:,ocols:]=data
            elif axis==0:
                nrows=orows+drows
                dset.resize(nrows,axis=0)
                dset[orows:,:]=data
            elif axis==1:
                ncols=ocols+dcols
                dset.resize(ncols,axis=1)
                dset[:,ocols:]=data


def write_array_h5(h5filename,groupname,dataname,data):
        with h5py.File(h5filename,mode='a') as tf:
            grppth='/'+groupname
            tpth = grppth+'/'+dataname
            dshape=data.shape
            if not grppth in tf:
                grp = tf.create_group(groupname)
            else:
                grp=tf[grppth]
            if data.dtype=='O':
                data=data.astype("str")
            if not tpth in tf:
                    dset = grp.create_dataset(tpth,dshape,
                                              chunks=True,
                                              maxshape=(None,)*len(dshape),
                                              compression=32001,
                                              compression_opts=(0, 0, 0, 0, 3, 2, 0),
                                              shuffle=True,
                                              data=data)
            else:
                dset=grp[dataname]
                oshape=dset.shape
                dset.resize(oshape[2]+dshape[2],2)
                dset[:,:,oshape[2]:]=data


print 'loading data'
negmat = h5py.File('neg_train.mat','r')
posmat = h5py.File('Noonan_peaks.h5','r')

X_neg = np.transpose(np.array(negmat['trainxdata']),axes=(2,0,1))
X_pos =np.transpose(np.array(posmat['trainxdata']),axes=(2,0,1))

X_mat = np.concatenate([X_neg,X_pos],axis=0)

y_neg= np.array(negmat['traindata'])
y_pos = np.array(posmat['traindata'])

y_mat = np.concatenate([y_neg,y_pos],axis=1).T

X_train, X_test, y_train, y_test = train_test_split(X_mat, y_mat, test_size=0.33, random_state=1337)

trainfile="Noonan_train.h5"
testfile="Noonan_test.h5"

write_array_h5(trainfile,"","trainxdata",X_train)

write_mat_h5(trainfile,"","traindata",y_train)

write_array_h5(testfile,"","testxdata",X_test)

write_mat_h5(testfile,"","testdata",y_test)
