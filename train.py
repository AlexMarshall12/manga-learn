import h5py
import scipy as sp
import boto3
import glob
import numpy as np
import random
import keras
import logging

from sklearn.externals import joblib
from skimage import color,io
from keras import backend as K
from keras.models import Sequential
from keras.utils.io_utils import HDF5Matrix
from keras.layers import Dense, Dropout, Activation, Flatten, normalization, convolutional
from keras.layers.convolutional import ZeroPadding2D
from keras.layers import Convolution2D, MaxPooling2D,Reshape
from keras.optimizers import SGD
from keras.optimizers import Adadelta

def Colorize(weights_path=None):
    model = Sequential()
    # input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
    # this applies 32 convolution filters of size 3x3 each.

    model.add(Convolution2D(256, 3, 3, border_mode='valid',input_shape=(384,50,50)))
    model.add(convolutional.ZeroPadding2D(padding=(1,1)))
    model.add(Activation('relu'))
    model.add(normalization.BatchNormalization())

    model.add(Convolution2D(112, 3, 3, border_mode='valid'))
    model.add(convolutional.ZeroPadding2D(padding=(1,1)))
    model.add(Activation('relu'))
    model.add(normalization.BatchNormalization())
   
    #softmax
    print model.summary()
    model.add(Reshape((112,50*50)))

    model.add(Activation('softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model

colorize = Colorize()
color_sgd = SGD(lr=0.001,momentum=0.9)
color_adadelta = Adadelta()
colorize.compile(optimizer=color_sgd,loss='categorical_crossentropy')
f = h5py.File("raw.h5","r")
dset_X = f.get('X')
dset_y = f.get('y')
print dset_X.shape
print dset_y.shape
mini_batch_size=1
epochs = 1

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        batch_loss = logs.get('loss')
        self.losses.append(batch_loss)

def generate_train_batch():
    while 1:
        for i in xrange(0,dset_X.shape[0],mini_batch_size):
            yield dset_X[i:i+mini_batch_size,:,:,:],dset_y[i:i+mini_batch_size,:,:]

history = LossHistory()
colorize.fit_generator(generate_train_batch(),samples_per_epoch=2,nb_epoch=2,callbacks=[history])


numpy_loss_history = np.array(history.losses)
np.savetxt("loss_history.txt",numpy_loss_history, delimiter = ",")

#for e in range(epochs):
#loss_history = history_callback.history["loss"]
#print history_callback
#    counter = 0
#    for mini_batch in range(1,10000,mini_batch_size):
#        X = dset_X_train[mini_batch:mini_batch+mini_batch_size,:,:,:]
#        y = dset_y_train[mini_batch:mini_batch+mini_batch_size,:,:]
#        print X.shape
#        print y.shape
#        colorize.train_on_batch(X,y)
#        counter+=1
#        print counter
#        if counter % 3 == 0:
#            colorize.save_weights('colorize_weights.h5',overwrite=True)
#            colorize.test_on_batch(dset_X_val[counter:counter+1,:,:,:],dset_y_val[counter:counter+1,:,:]),
