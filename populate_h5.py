import h5py
import tables
from itertools import izip_longest
import glob
import boto3
import numpy as np
import scipy as sp
from PIL import Image

from sklearn.externals import joblib
from skimage import color,io

from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.optimizers import SGD


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


KNN = joblib.load('KNN.pkl')
s3 = boto3.resource('s3')

files = glob.glob('../manga-resized/sliced/*.png')
model = VGG_16()
sgd = SGD(lr=0.1,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(optimizer=sgd,loss='categorical_crossentropy')

#h5file = open_file("tutorial1.h5", mode = "w", title = "Test file")

import tables

filters = tables.Filters(complevel=5, complib='zlib')

with tables.openFile('raw.h5','w') as f:
#    filters = tables.Filters(complib='blosc', complevel=5)
    dset_X = f.create_earray(f.root, 'X', tables.Atom.from_dtype(np.dtype('Float64')), (0,960,224,224),filters=filters)
    set_y = f.create_earray(f.root, 'y', tables.Atom.from_dtype(np.dtype('Float64')), (0,112,224*224),filters=filters)
    for fl in files[0:12000]:
        img = Image.open(fl)
        if img.getbbox():
            X_chunk,y_chunk=get_arrays(fl)
            print X_chunk.dtype
            dset_X.append(X_chunk)
            dset_y.append(y_chunk)
        else:
            print "was black"
            pass
    

#n_images = 0
#
#with h5py.File('image1.h5','w') as f: 
#    dset_X = f.create_dataset('X',(1,960,224,224),maxshape=(None,960,224,224),chunks=True,compression='gzip', compression_opts=5)
#    dset_y = f.create_dataset('y',(1,112,224*224),maxshape=(None,112,224*224),chunks=True,compression='gzip', compression_opts=5)
#    n_images = 0
#    for fl in files[200:236]:
#        X_chunk,y_chunk = get_arrays(fl) 
#        dset_X.resize(n_images+1,axis=0,)
#        dset_y.resize(n_images+1,axis=0,)
#        print dset_X.compression,dset_y.compression
#        print dset_X.shape,dset_y.shape
#        dset_X[n_images:n_images+1,:,:,:]=X_chunk
#        dset_y[n_images:n_images+1,:,:]=y_chunk
#        n_images+=1
#    #s3.Bucket('my-bucket').put_object(Key='test.jpg', Body=f)
    
