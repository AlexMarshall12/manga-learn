import h5py
import scipy.misc
import numpy as np

from keras import backend as K
from keras.models import Sequential
from keras.utils.io_utils import HDF5Matrix
from keras.layers import Dense, Dropout, Activation, Flatten, normalization, convolutional
from keras.layers.convolutional import ZeroPadding2D
from keras.layers import Convolution2D, MaxPooling2D,Reshape
from keras.optimizers import SGD

def extract_hypercolumn(model, layer_indexes, instance):
    layers = [model.layers[li].output for li in layer_indexes]
    get_feature = K.function([model.layers[0].input],layers)
    assert instance.shape == (1,3,224,224)
    feature_maps = get_feature([instance])
    hypercolumns = []
    for convmap in feature_maps:
        for fmap in convmap[0]:
            upscaled = scipy.misc.imresize(fmap, size=(224, 224),
                                        mode="F", interp='bilinear')
            hypercolumns.append(upscaled)

    return np.asarray(hypercolumns)

#def generate_batch_from_hdf5():
#    f = h5py.File("raw_image_tensors.h5","r")
#    dset_X = f.get('X')
#    dset_y = f.get('y')
#     
#    print dset_X.shape,dset_y.shape
#    total_data_size=dset_X.shape[0]
#    for i in range(dset_X.shape[0]):
#        X = dset_X[i:i+1,:,:,:]
#        print "1 X instance from hdf5 file has a shape of: ",X.shape
#        X = np.tile(X,(1,3,1,1))    #triplicate L values to feed into VGG16
#        hc = extract_hypercolumn(model,[3,8,15,22],X)
#        hc = np.expand_dims(hc,axis=0)
#        print "hc has a shape of: ",hc.shape
#        yield dsey_X[i:i+1,:,:,:],dset_y[i:i+1,:,:]
def generate_batch_from_hdf5():
    f = h5py.File("raw_image_tensors.h5","r")
    dset_X = f.get('X')
    dset_y = f.get('y')
     
    print dset_X.shape,dset_y.shape
    for i in range(dset_X.shape[0]):
        X_n = dset_X[i:i+1,:,:,:]
        X_t = np.tile(X_n,(1,3,1,1))
        hc = extract_hypercolumn(model,[3,8],X_t)
        yield 5,6

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
sgd = SGD(lr=0.1,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(optimizer=sgd,loss='categorical_crossentropy')

#for fl in files: 
#    out = model.predict(color.rgb2lab(fl))
#    hc = extract_hypercolumn(model, layers_extract = [3,8])
color = Colorize()
color.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=["accuracy"])
color.fit_generator(generate_batch_from_hdf5(),samples_per_epoch=5,nb_epoch=5)
