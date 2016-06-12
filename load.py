import h5py
import scipy as sp
import numpy as np

from keras import backend as K
from keras.models import Sequential
from keras.utils.io_utils import HDF5Matrix
from keras.layers import Dense, Dropout, Activation, Flatten, normalization, convolutional
from keras.layers.convolutional import ZeroPadding2D
from keras.layers import Convolution2D, MaxPooling2D,Reshape
from keras.optimizers import SGD


def generate_batch_from_hdf5():
    f = h5py.File("raw_tensors.h5","r")
    dset_X = f.get('X')
    dset_y = f.get('y')
    print "were doing it"
     
    print "these are the shapes",dset_X.shape,dset_y.shape
    total_data_size=dset_X.shape[0]
    for i in range(dset_X.shape[0]):
        print "X has a shape of: ",dset_X[i:i+1,:,:,:].shape
        print "y has a shape of: ",dset_y[i:i+1,:,:].shape
        arr_X = np.asarray(dset_X[i:i+1,:,:,:])
        print "arr_X",arr_X.shape
        yield dset_X[i:i+1,:,:,:],dset_y[i:i+1,:,:]

def Colorize(weights_path=None):
    model = Sequential()
    # input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
    # this applies 32 convolution filters of size 3x3 each.

    model.add(Convolution2D(512, 1, 1, border_mode='valid',input_shape=(960,224,224)))
    model.add(Activation('relu'))
    model.add(normalization.BatchNormalization())

    model.add(Convolution2D(256, 1, 1, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(normalization.BatchNormalization())

    model.add(Convolution2D(112, 1, 1, border_mode='valid'))
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

model = Colorize()
sgd = SGD(lr=0.1,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=["accuracy"])

#for fl in files: 
#    out = model.predict(color.rgb2lab(fl))
#    hc = extract_hypercolumn(model, layers_extract = [3,8])

model.fit_generator(generate_batch_from_hdf5(),samples_per_epoch=1000,nb_epoch=5,validation_data=(X_val,y_val))

score = model.evaluate(X_test,y_test)
