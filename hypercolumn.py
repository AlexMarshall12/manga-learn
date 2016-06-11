from matplotlib import pyplot as plt
import glob
import theano
import os
import numpy as np
import scipy as sp
import pickle
import psutil

from keras.models import Sequential
from keras.utils.io_utils import HDF5Matrix
from keras.layers import Dense, Dropout, Activation, Flatten, normalization, convolutional
from keras.layers import Convolution2D, MaxPooling2D,Reshape
from keras.optimizers import SGD

from skimage import color,io
from sklearn.manifold import TSNE
from sklearn import manifold
from sklearn import cluster
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib

KNN = joblib.load('KNN.pkl')

def load_hdf5():
    X_train = HDF5Matrix('raw_arrays.h5','X')
    y_train = HDF5Matrix('raw_arrays.h5','y') 
    return X_train,y_train
    
def cache_data(data,path):
    if os.path.isdir(os.path.dirname(path)):
        file = open(path,'wb')
        pickle.dump(data,file)
    else:
        print 'Directory doesnt exist'

def restore_data(path):
    data = dict()
    if os.path.isfile(path):
        file = open(path,'rb')
        data = pickle.load(file)
    return data

def VGG_16(weights_path=None):
    model = Sequential()
    # input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
    # this applies 32 convolution filters of size 3x3 each.

    model.add(Convolution2D(64, 3, 3, border_mode='valid',input_shape=(1,224,224)))
    model.add(convolutional.ZeroPadding2D(padding=(1,1)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(convolutional.ZeroPadding2D(padding=(1,1)))
    model.add(Activation('relu'))
    model.add(normalization.BatchNormalization())

    model.add(Convolution2D(128, 3, 3, border_mode='valid'))
    model.add(convolutional.ZeroPadding2D(padding=(1,1)))
    model.add(Activation('relu'))
    model.add(Convolution2D(128, 3, 3))
    model.add(convolutional.ZeroPadding2D(padding=(1,1)))
    model.add(Activation('relu'))
    model.add(normalization.BatchNormalization())

    model.add(Convolution2D(256, 3, 3, border_mode='valid'))
    model.add(convolutional.ZeroPadding2D(padding=(1,1)))
    model.add(Activation('relu'))
    model.add(Convolution2D(256, 3, 3, border_mode='valid'))
    model.add(convolutional.ZeroPadding2D(padding=(1,1)))
    model.add(Activation('relu'))
    model.add(Convolution2D(256, 3, 3, border_mode='valid'))
    model.add(convolutional.ZeroPadding2D(padding=(1,1)))
    model.add(Activation('relu'))
    model.add(normalization.BatchNormalization())
    
    model.add(Convolution2D(322, 3, 3, border_mode='valid'))
    model.add(convolutional.ZeroPadding2D(padding=(1,1)))
    model.add(Activation('relu'))
    model.add(Convolution2D(322, 3, 3, border_mode='valid'))
    model.add(convolutional.ZeroPadding2D(padding=(1,1)))
    model.add(Activation('relu'))
    model.add(Convolution2D(322, 3, 3, border_mode='valid'))
    model.add(convolutional.ZeroPadding2D(padding=(1,1)))
    model.add(Activation('relu'))
    model.add(normalization.BatchNormalization())

    #softmax
    model.add(Reshape((322,224*224)))
    model.add(Activation('softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model
#
#cache_path = os.path.join('cache','data.dat')
#if not os.path.isfile(cache_path):
#    X,y = populate()
#    cache_data((X,y),cache_path)
#else:
#    print "Restore data from cache!"
#    X,y = restore_data(cache_path)
#    print "X shape: ",X.shape,"y shape: ",y.shape
#    print "data restored!"
#print "X shape: ",X.shape
#print "y shape: ",y.shape

X = HDF5Matrix('raw_arrays.h5','X')
y = HDF5Matrix('raw_arrays.h5','y')

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = VGG_16()
sgd = SGD(lr=0.1,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=["accuracy"])

model.fit(X_train,y_train,batch_size=32,shuffle='batch',show_accuracy=True)

#
#for e in range(nb_epoch):
#    print("epoch %d" % e)
#    for X_train, Y_train in BatchGenerator(): 
#        model.fit(X_batch, Y_batch, batch_size=32, nb_epoch=1)
#
#model.fit(X_train,y_train,batch_size=32,nb_epoch=1,show_accuracy=True,verbose=1,validation_data=(X_test,y_test))
#
#y_pred = model.predict(X_test)
#y_pred_reshaped = np.reshape(y_pred,(y_pred.shape[2],y_pred.shape[1]))
#cat_pred = np.argmax(y_pred_reshaped,axis=1)
#
#y_test_reshaped = np.reshape(y_test,(y_test.shape[2],y_test.shape[1]))
#cat_test = np.argmax(y_test_reshaped,axis=1)
#
#correct_pixel_percentage = 1 - (np.count_nonzero(cat_pred-cat_test))/cat_pred.shape[0]
#print "percentage of correct pixel guesses: ",correct_pixel_percentage
