from classifieurs.classifiers import *
from skimage.feature import hog
from skimage import io, util
import numpy as np
import glob
import cv2

X_data = []
Y_data = []

print("Importing data...")

# Import positive images
for filename in glob.glob("dataset-premierclassifieur/images_transfomes/images_pretes_a_etre_utliser/pos/*.jpg"):
    I = io.imread(filename)
    if (I.shape[0] == 240 and I.shape[1] == 160):
        I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
        hog_image = hog(I, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=False)
    if (len(hog_image) == 1200):
        X_data.append(hog_image)
        Y_data.append(1)

# Import negative images
for filename in glob.glob("dataset-premierclassifieur/images_transfomes/images_pretes_a_etre_utliser/neg/*.jpg"):
    I = io.imread(filename)
    if (I.shape[0] == 240 and I.shape[1] == 160):
        I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
        hog_image = hog(I, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=False)
        X_data.append(hog_image)
        Y_data.append(-1)

X_data = np.array(X_data)
Y_data = np.array(Y_data)

print("Data imported.")

# Train AdaBoost classifier

print("Training AdaBoost classifier...")

classifier = AdaBoost(400)
classifier.train(X_data, Y_data)

print("AdaBoost classifier trained.")
