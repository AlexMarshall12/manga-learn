import h5py
from itertools import izip_longest
import glob
import numpy as np
from sklearn.externals import joblib
from skimage import color,io
KNN = joblib.load('KNN.pkl')
def loadfunc(files):
    for fl in files:
        print fl
        img = color.rgb2lab(io.imread(fl)[..., :3])
        yield img

def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return izip_longest(*args, fillvalue=fillvalue)

batch_size=5
files = glob.glob('../manga-resized/sliced_images/*.png')
f = h5py.File('raw_arrays.h5','w')
dset_X = f.create_dataset('X',(1,1),maxshape=(None,1),chunks=True)
dset_y = f.create_dataset('y',(1,112),maxshape=(None,112),chunks=True)

n_images = 0
for i in grouper(files,batch_size):
    z = np.stack(loadfunc(list(i)),axis=-1)
    X = z[..., :1,:]
    y = z[..., 1:,:]
    print "y shape: ",y.shape
    print "X shape: ",X.shape
    X_rows,X_cols,X_channels,X_samples=X.shape
    y_rows,y_cols,y_channels,y_samples=y.shape
    X_chunk = np.reshape(X,(X_samples*X_rows*X_cols,X_channels))
    y_chunk = np.reshape(y,(y_samples*y_rows*y_cols,y_channels))
    classed = KNN.predict_proba(y_chunk)
    dset_X.resize(n_images+224*224*5,axis=0)
    dset_y.resize(n_images+224*224*5,axis=0)
    print "X_chunk: ",X_chunk.shape,"dset_X: ",dset_X.shape
    print "y_chunk: ",classed.shape,"dset_y: ",dset_y.shape
    dset_X[n_images:n_images+224*224*5,:]=X_chunk
    dset_y[n_images:n_images+224*224*5,:]=classed
    n_images+= 5*224*224
    print dset_X.shape
    print dset_y.shape
