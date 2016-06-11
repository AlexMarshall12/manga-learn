import glob
import pickle
import os
import math
import pandas as pd
import datetime
import theano
import theano.tensor as T
from skimage import color,io,transform
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import NearestNeighbors
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import model_from_json
from sklearn.metrics import log_loss
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from sklearn.externals import joblib
import scipy.ndimage, scipy.misc
import numpy as np
import sys

kmeans = joblib.load('kmeans.pkl')
kde = joblib.load('kde.pkl')
model_from_cache = 0

def load_data():
    filenames = glob.glob('manga-resized/*.jpg')
    L = []
    ab = []
    for fn in filenames:
        image = color.rgb2lab(io.imread(fn))    
        for i in range(2*(num_rows/100-1)):
            for j in range(2*(num_columns/100-1)):
                image_square_L = image[50*i:50*i+100,50*j:50*j+100,:1]
                image_square_ab = image[50*i:50*i+100,50*j:50*j+100,1:]
                w,h,d = image_square_ab.shape
                reshaped_L = np.reshape(image_square_L,(w*h,d))
                stacked_ab = np.reshape(image_square_ab,(w*h,d))
                print "stacked_ab shape",stacked_ab.shape
                targets = kmeans.predict(stacked_ab)
                print "predicted_kmeans shape",targets.shape
                L.append(reshaped_L)
                ab.append(targets)

    X = np.asarray(L)
    y = np.asarray(ab)
    return X,y

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

def save_model(model):
    json_string = model.to_json()
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    open(os.path.join('cache','architecture.json'),'w').write(json_string)
    model.save_weights(os.path.join('cache','model_weights.h5'),overwrite=True)

def read_model():
    model = model_from_json(open(os.path.join('cache','architecture.json')).read())
    model.load_weights(os.path.join('cache','model_weights.h5'))
    model_from_cache = 1
    return model

def custom_loss(y_true,y_pred):
    factor = -1*kde.score(y_pred)
    if factor >= 0:
        factor = 1
    cce = T.nnet.categorical_crossentropy(y_pred,y_true)*(1/factor)
    return cce

cache_path = os.path.join('cache','data.dat')
if not os.path.isfile(cache_path):
    X,y = load_data()
    cache_data((X,y),cache_path)
else:
    print "Restore data from cache!"
    X,y = restore_data(cache_path)
    print "X shape: ",X.shape,"y shape: ",y.shape
    print "data restored!"

batch_size = 64
nb_classes = 313
np_epoch = 2
img_rows, img_cols = 120,78
num_rows = 1200
num_columns = 800
X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.2, random_state=42)

if model_from_cache ==1:
    print 'reading model from cache'
    model = read_model()
    model.compile(loss='categorical_crossentropy',optimizer='adadelta')
    score = model.evaluate(X_test,y_test,show_accuracy=True,verbose=0)
    print 'Score: ',score
else:     
    model = Sequential()
    model.add(Convolution1D(32, 3, 3, border_mode='valid', input_shape=(3,10000)))
    model.add(Activation('relu'))
    model.add(Convolution1d(32,3,3))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(Convolution1D(64,3,3,border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution1D(64,3,3))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(10))
    model.add(TimeDistributed('softmax'))
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',optimizer=sgd)
    
    model.fit(X_train,y_train,batch_size=32,nb_epoch=1,show_accuracy=True,verbose=1,validation_data=(X_test,y_test))
    score = model.evaluate(X_test,y_test,show_accuracy=True,verbose=0)
    
    save_model(model)

