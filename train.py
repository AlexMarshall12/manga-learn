import h5py
import scipy as sp
import boto3
import glob
import numpy as np
import random
import keras

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

    model.add(Convolution2D(512, 1, 1, border_mode='valid',input_shape=(960,224,224)))
    model.add(Activation('relu'))
    model.add(normalization.BatchNormalization())

    model.add(Convolution2D(256, 1, 1, border_mode='valid'))
#    model.add(convolutional.ZeroPadding2D(padding=(1,1)))
    model.add(Activation('relu'))
    model.add(normalization.BatchNormalization())

    model.add(Convolution2D(112, 1, 1, border_mode='valid'))
 #   model.add(convolutional.ZeroPadding2D(padding=(1,1)))
    model.add(Activation('relu'))
    model.add(normalization.BatchNormalization())
   
    print "output shape: ",model.output_shape
    #softmax
    model.add(Reshape((112,224*224)))

    print "output_shape after reshaped: ",model.output_shape
    model.add(Activation('softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

colorize = Colorize()
color_sgd = SGD(lr=0.001)
color_adadelta = Adadelta()
colorize.compile(optimizer=color_sgd,loss='categorical_crossentropy',metrics=["accuracy","val_loss"])

f = h5py.File("raw.h5","r")
dset_X = f.get('X')
dset_y = f.get('y')
mini_batch_size=4
epochs = 1
history = LossHistory()

for e in range(epochs):
    counter = 0
    for mini_batch in range(1,dset_X.shape[0],mini_batch_size):
        X = dset_X[mini_batch:mini_batch+mini_batch_size,:,:,:]
        y = dset_y[mini_batch:mini_batch+mini_batch_size,:,:]
        X_train = X[:3,:,:,:]
        y_train = y[:3,:,:]
        X_val = X[3:,:,:,:]
        y_val = y[3:,:,:]
        colorize.fit(X_train,y_train,batch_size=mini_batch_size,nb_epoch=1,callbacks=[history],validation_data = (X_val,y_val))
        counter+=1
        print counter
        if counter % 100 == 0:
            colorize.save_weights('colorize_weights.h5',overwrite=True)
with open('log.txt','wb') as f:
    f.write(history.losses)
       
