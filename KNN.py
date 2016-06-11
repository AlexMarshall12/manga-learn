import glob
from skimage import color,io
from sklearn.utils import shuffle
#from sklearn.cluster import KMeans
from sklearn.externals import joblib
#from sklearn.neighbors.kde import KernelDensity
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import seaborn as sns
import pandas as pd

filenames = glob.glob('manga-resized/*.jpg')
images = []

for fn in filenames:
    image = color.rgb2lab(io.imread(fn))
    ab = image[:,:,1:]
    width,height,d = ab.shape
    reshaped_ab = np.reshape(ab,(width*height,d))
    print reshaped_ab.shape
    images.append(reshaped_ab)

all_abs = np.vstack(images)
all_abs = shuffle(all_abs,random_state=0)

df = pd.DataFrame(all_abs[:3000],columns=["a","b"])

top_a,top_b = df.max()
bottom_a,bottom_b = df.min()
x = np.linspace(bottom_a,top_a,(top_a-bottom_a)/12.5)
y = np.linspace(bottom_b,top_b,(top_b-bottom_b)/12.5)

print "x shape: ",x.shape,"y shape: ",y.shape

xv,yv = np.meshgrid(x,y)
D = np.stack((xv,yv),2)

L = D.shape[0]*D.shape[1]
t = np.linspace(0,L-1,L)
D = np.vstack(D)
print "D shape: ",D.shape,"t shape: ",t.shape

KNN = KNeighborsClassifier(n_neighbors = 10,weights='distance')
KNN.fit(D,t)

#sns.jointplot(x="a",y="b",data=df)
#kde = KernelDensity(kernel = 'gaussian',bandwidth=0.2).fit(all_abs[:3000])
#kmeans = KMeans(n_clusters = n_colors,random_state=0).fit(all_abs[:3000])
#joblib.dump(kmeans,'kmeans.pkl')
joblib.dump(KNN,'KNN.pkl')
#joblib.dump(kde,'kde.pkl')
