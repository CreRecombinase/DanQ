import itertools
import gzip
from sklearn import preprocessing
from itertools import groupby
from itertools import zip_longest
import numpy as np
import csv
import tables
import h5py
import pandas as pd
import scipy.io

def fastest_fasta(fasta_name):
    with open(fasta_name,'rt') as f:
        for line in itertools.islice(f, 1, None, 2):
            yield(line.rstrip('\n'))

def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


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
                                              data=data)
            else:
                dset=grp[dataname]
                oshape=dset.shape
                dset.resize(oshape[2]+dshape[2],2)
                dset[:,:,oshape[2]:]=data

    
def read_array_index(h5filename,groupname,dataname,index):
    with h5py.File(h5filename,mode='r') as tf:
        grp=tf['/'+groupname]
        data=grp[dataname]
        index=index[index != np.array(None)].astype('int64')
        print(index.dtype)
        min_ind=index.min()
        max_ind=index.max()+1
        n_ind= index-min_ind
        print("Reading from"+str(min_ind)+":"+str(max_ind))
        tdata=data[:,:,min_ind:max_ind]
        return tdata[:,:,n_ind]
