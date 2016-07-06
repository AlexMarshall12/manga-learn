import h5py
import scipy as sp
import boto3
import glob
import numpy as np
import random

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

KNN = joblib.load('KNN.pkl')

def extract_hypercolumn(model, layer_indexes, instance):
    layers = [model.layers[li].output for li in layer_indexes]
    get_feature = K.function([model.layers[0].input],layers)
    assert instance.shape == (1,3,224,224)
    feature_maps = get_feature([instance])
    hypercolumns = []
    for convmap in feature_maps:
        for fmap in convmap[0]:
            upscaled = sp.misc.imresize(fmap, size=(224, 224),
                                        mode="F", interp='bilinear')
            hypercolumns.append(upscaled)

    return np.asarray(hypercolumns)

def get_arrays(each_file):
    lab = color.rgb2lab(io.imread(each_file))
    X = lab[:,:,:1]
    y = lab[:,:,1:]
    X_rows,X_columns,X_channels=X.shape
    y_rows,y_columns,y_channels=y.shape
    X_channels_first = np.transpose(X,(2,0,1))
    X_sample = np.expand_dims(X_channels_first,axis=0)
    X_3d = np.tile(X_sample,(1,3,1,1))
    hc = extract_hypercolumn(model,[3,8,15,22],X_3d)
    hc_expand_dims = np.expand_dims(hc,axis=0)
    y_reshaped = np.reshape(y,(y_rows*y_columns,y_channels))
    classed_pixels_first = KNN.predict_proba(y_reshaped)
    classed_classes_first = np.transpose(classed_pixels_first,(1,0))
    classed_expand_dims = np.expand_dims(classed_classes_first,axis=0)
    print "hypercolumn shape: ",hc_expand_dims.shape,"classified output color shape: ",classed_expand_dims.shape
    return hc_expand_dims,classed_expand_dims

def generate_batch():
    files = glob.glob('../manga-resized/sliced/*.png')
    random.shuffle(files)
    counter = 0
    while True:
        for fl in files: 
            counter+=1
            print counter
            if counter % 1000 == 0:
                colorize.save_weights('colorize_weights.h5',overwrite=True)
            yield get_arrays(fl)

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

def VGG_16(weights_path='vgg16_weights.h5'):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model


model = VGG_16()
sgd = SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(optimizer=sgd,loss='categorical_crossentropy')

colorize = Colorize()
color_sgd = SGD(lr=0.0001)
color_adadelta = Adadelta()
colorize.compile(optimizer=color_adadelta,loss='categorical_crossentropy',metrics=["accuracy"])

#from itertools import islice
#print list(islice(generate_batch(), 3))
#colorize.fit_generator(generate_batch(),samples_per_epoch=1,nb_epoch=5)
#for a, b in generate_batch(): print(a, b)

colorize.fit_generator(generate_batch(),samples_per_epoch=20,nb_epoch=1)
