import h5py
import scipy as sp
from itertools import izip_longest
import glob
import numpy as np
from sklearn.externals import joblib
from skimage import color,io

from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.optimizers import SGD

KNN = joblib.load('KNN.pkl')



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

files = glob.glob('../manga-resized/sliced_images/*.png')

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

with h5py.File('raw_tensors.h5','w') as f:
    dset_X = f.create_dataset('X',(1,960,224,224),maxshape=(None,960,224,224),chunks=True)
    dset_y = f.create_dataset('y',(1,112,224*224),maxshape=(None,112,224*224),chunks=True)
    n_images = 0
    for fl in files[:500]:
        img = color.rgb2lab(io.imread(fl)[..., :3])
        X = img[:,:,:1]
        y = img[:,:,1:]
        print "y shape: ",y.shape
        print "X shape: ",X.shape
        X_rows,X_columns,X_channels=X.shape
        y_rows,y_columns,y_channels=y.shape
        X_chunk = np.transpose(X,(2,0,1))
        X_chunk_3d = np.tile(X_chunk,(3,1,1))
        print "X_chunk_3d: ",X_chunk_3d.shape
        X_chunk_4d = np.expand_dims(X_chunk_3d,axis=0)
        print "X_chunk_4d: ",X_chunk_4d.shape
        hc = extract_hypercolumn(model,[3,8,15,22],X_chunk_4d)
        y_chunk = np.reshape(y,(y_rows*y_columns,y_channels))
        classed = KNN.predict_proba(y_chunk)
        classed = np.transpose(classed,(1,0))
        dset_X.resize(n_images+1,axis=0)
        dset_y.resize(n_images+1,axis=0)
        print "X_chunk: ",X_chunk.shape,"dset_X: ",dset_X.shape
        print "hypercolumn shape: ",hc.shape
        print "y_chunk: ",classed.shape,"dset_y: ",dset_y.shape
        dset_X[n_images:n_images+1,:,:,:]=hc
        dset_y[n_images:n_images+1,:,:]=classed
        n_images+= 1
        print dset_X.shape
        print dset_y.shape
    
