from classifieurs.classifiers import *
from skimage.feature import hog
from skimage import io, util
from random import shuffle
import numpy as np
import glob
import cv2
import shutil
import os

class Rectangle:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w 
        self.h = h
    
    def to_csv_line(self):
        return f"{self.y},{self.x},{self.h},{self.w}"

class Rectangles:
    def __init__(self, filename):
        self.rectangles = []
        self.filename = filename
    
    def add(self, rect):
        self.rectangles.append(rect)
    
    def get(self, index):
        return self.rectangles[index]
    
    def size(self):
        return len(self.rectangles)
    
    def import_rectangles(self):
        bbox = np.loadtxt(os.path.join("dataset-original/train/labels_csv", self.filename + ".csv"), delimiter=",")
        if bbox.ndim == 1:
            bbox = [bbox]
        
        for i in range(len(bbox)):
            self.add(Rectangle(bbox[i][1], bbox[i][0], bbox[i][3], bbox[i][2]))

def IoU(rect1, rect2):
    xA = max(rect1.x, rect2.x)
    yA = max(rect1.y, rect2.y)
    xB = min(rect1.x + rect1.w, rect2.x + rect2.w)
    yB = min(rect1.y + rect1.h, rect2.y + rect2.h)

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = rect1.w * rect1.h
    boxBArea = rect2.w * rect2.h

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

def filtre_nms (rectangles, scores, tresh_iou = 0.5) :
    # trier les boites par score de confiance decroissant
    rectangles = rectangles[scores.argsort()[:: -1]]
    # liste des boites que l'on garde pour cette image a l'issue du NMS
    results = np.empty ((0 ,2) )
    # on garde forcement la premiere boite , la plus sure
    results = np.vstack ((results, [rectangles [0],scores[0]]) )
    # pour toutes les boites suivantes , les comparer a celles que l'on garde
    for n in range (1, len (rectangles)):
        for m in range (len(results)):
            if IoU (rectangles[n] , results[m][0]) > tresh_iou :
                # recouvrement important ,
                # et la boite de results_out_i a forcement un score plus haut
                break
            elif m == len (results) - 1 :
                # c'etait le dernier test pour verifier si cette detection est a conserver
                results = np.vstack((results, [rectangles[n], scores[n]]))

    return results

###############################

def copy_dataset_to_new_folder():
    if os.path.exists('dataset-classifieur-ameliore') :
        shutil.rmtree('dataset-classifieur-ameliore')
    os.makedirs('dataset-classifieur-ameliore/pos')
    os.makedirs('dataset-classifieur-ameliore/neg')
    for filename in glob.glob("dataset-premierclassifieur/images_transfomes/images_pretes_a_etre_utliser/pos/*.jpg"):
        shutil.copyfile(filename, "dataset-classifieur-ameliore/pos/" + filename.split("\\")[-1])

    for filename in glob.glob("dataset-premierclassifieur/images_transfomes/images_pretes_a_etre_utliser/neg/*.jpg"):
        shutil.copyfile(filename, "dataset-classifieur-ameliore/neg/" + filename.split("\\")[-1])

def get_trained_classifier():
    X_data = []
    Y_data = []

    # Import positive images
    for filename in glob.glob("dataset-classifieur-ameliore/pos/*.jpg"):
        I = io.imread(filename)
        if (I.shape[0] == 240 and I.shape[1] == 160):
            I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
            hog_image = hog(I, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=False)
        if (len(hog_image) == 1200):
            X_data.append(hog_image)
            Y_data.append(1)

    # Import negative images
    for filename in glob.glob("dataset-classifieur-ameliore/neg/*.jpg"):
        I = io.imread(filename)
        if (I.shape[0] == 240 and I.shape[1] == 160):
            I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
            hog_image = hog(I, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=False)
            X_data.append(hog_image)
            Y_data.append(-1)

    X_data = np.array(X_data)
    Y_data = np.array(Y_data)

    # Train AdaBoost classifier

    classifier = AdaBoost(400)
    classifier.train(X_data, Y_data)

    return classifier

def import_test_data():
    X_test_data = []
    test_data_filenames = []

    for filename in glob.glob("dataset-fenetre_glissante/crop_fenetre_a_classifier/*.jpg"):
        I = io.imread(filename)
        if (I.shape[0] == 240 and I.shape[1] == 160):
            I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
            hog_image = hog(I, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=False)
            X_test_data.append(hog_image)
            test_data_filenames.append(filename)

    X_test_data = np.array(X_test_data)
    test_data_filenames = np.array(test_data_filenames)

    return X_test_data, test_data_filenames

def get_positive(classifier, X_test_data, test_data_filenames):
    # Predict test data

    Y_test_data = classifier.predict(X_test_data)

    positive_images_filenames = test_data_filenames[Y_test_data == 1]
    
    scores = classifier.prediction_scores(X_test_data)
    scores = scores[Y_test_data == 1]

    return positive_images_filenames, scores

def import_real_rectangles():
    rectangles = []
    for label in os.listdir("dataset-original/train/labels_csv"):
        rectangles.append(Rectangles(label.split(".")[0]))
        rectangles[-1].import_rectangles()
    return rectangles

def get_false_positive_rectangles(positive_images_filenames, scores, real_rectangles):
    false_positive_rectangles = []
    positive_images_filenames = positive_images_filenames[np.argsort(scores)[::-1]]
    for filename in positive_images_filenames:
        filename_nopath_noextention = filename.split("\\")[-1][:-4]

        data = filename_nopath_noextention.split("-")
        # data is [filename, y, x, h, w]
        
        w = int(float(data[4]))/1.1
        h = int(float(data[3]))/1.1
        x = int(float(data[2])) + w*0.1
        y = int(float(data[1])) + h*0.1
        detected = Rectangle(x,y,w,h)
        
        is_real = False
        
        # Find the Rectangles object corresponding to the image
        for rectangles in real_rectangles:
            if rectangles.filename == data[0]:
                # Find the rectangle corresponding to the image
                for i in range(rectangles.size()):
                    rect = rectangles.get(i)
                    if IoU(rect, detected) > 0.0:
                        is_real = True
                        break
        
        if is_real == False:
            false_positive_rectangles.append(filename)

    return false_positive_rectangles

def copy_false_positive_images(false_positive_rectangles):
    for filename in false_positive_rectangles:
        shutil.copyfile(filename, "dataset-classifieur-ameliore/neg/false_positive_" + filename.split("\\")[-1])