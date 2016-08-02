import pandas as pd 
from sklearn import linear_model
import matplotlib.colors as colors
import xml.etree.ElementTree as ET
import numpy as np
from svg_parse import compute_features
from sklearn.cross_validation import train_test_split

X = pd.read_csv('features/train_features.txt',header=0,names=['area','length','complexity','lines','curves'])
y = pd.read_csv('labels/train_labels.txt',header=0,names=['r','g','b'])
X_norm = X.apply(lambda X: (X - np.mean(X)) / (np.max(X) - np.min(X)))   #normalize features
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.1, random_state=42)

l_model = linear_model.LinearRegression()
l_model.fit(X_train,y_train)

#X_test = pd.read_csv('data/test_features.txt',header=0,names=['area','length','index','complexity','lines','curves','intensity'])

with open('test_svgs/test.svg','rb') as o:
    tree = ET.parse(o)
    root = tree.getroot()
    for child in root:
        features = compute_features(child.attrib['d'])
        print l_model.predict(features)
        hex_guess = colors.rgb2hex(tuple(np.squeeze(l_model.predict(features))))
        child.set('fill',hex_guess)
    tree.write('best_guess.svg')

print l_model.score(X_test, y_test)
