import h5py
from itertools import izip_longest
import glob
import numpy as np
from sklearn.externals import joblib
from skimage import color,io
KNN = joblib.load('KNN.pkl')


files = glob.glob('../manga-resized/sliced_images/*.png')

with h5py.File('raw_image_tensors.h5','w') as f:
    dset_X = f.create_dataset('X',(1,1,224,224),maxshape=(None,1,224,224),chunks=True)
    dset_y = f.create_dataset('y',(1,112,224*224),maxshape=(None,112,224*224),chunks=True)
    n_images = 0
    for fl in files[:1000]:
        img = color.rgb2lab(io.imread(fl)[..., :3])
        X = img[:,:,:1]
        y = img[:,:,1:]
        print "y shape: ",y.shape
        print "X shape: ",X.shape
        X_rows,X_columns,X_channels=X.shape
        y_rows,y_columns,y_channels=y.shape
        X_chunk = np.transpose(X,(2,0,1))
        y_chunk = np.reshape(y,(y_rows*y_columns,y_channels))
        classed = KNN.predict_proba(y_chunk)
        classed = np.transpose(classed,(1,0))
        dset_X.resize(n_images+1,axis=0)
        dset_y.resize(n_images+1,axis=0)
        print "X_chunk: ",X_chunk.shape,"dset_X: ",dset_X.shape
        print "y_chunk: ",classed.shape,"dset_y: ",dset_y.shape
        dset_X[n_images:n_images+1,:,:,:]=X_chunk
        dset_y[n_images:n_images+1,:,:]=classed
        n_images+= 1
        print dset_X.shape
        print dset_y.shape
