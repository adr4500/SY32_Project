from classifieurs.classifiers import *
from operation_sur_images.fenetre_glissantes import windows
from operation_sur_images.resize import label_image
from skimage.feature import hog
from skimage import io, util
import numpy as np
import glob
import cv2
import os
import shutil

X_data = []
Y_data = []



def gen_data():
    
    label_image.crop_and_save_images("dataset-original/train/labels_csv","dataset-original/train/images/pos","dataset-premierclassifieur/images_transfomes/crop/pos")
    
    label_image.crop_to_ratio('dataset-premierclassifieur/images_transfomes/crop/pos')
    
    label_image.resize_images('dataset-premierclassifieur/images_transfomes/crop/pos/cropped',"dataset-premierclassifieur/images_transfomes/images_pretes_a_etre_utliser/pos",(160,240))

    label_image.flip_images('dataset-premierclassifieur/images_transfomes/images_pretes_a_etre_utliser/pos')
    
    label_image.random_crop_images("dataset-premierclassifieur/images/neg", "dataset-premierclassifieur/images_transfomes/crop/neg")

    label_image.resize_images('dataset-premierclassifieur/images_transfomes/crop/neg','dataset-premierclassifieur/images_transfomes/images_pretes_a_etre_utliser/neg',(160,240))
       
    label_image.flip_images('dataset-premierclassifieur/images_transfomes/images_pretes_a_etre_utliser/neg')

def windows_cropping(path_imput_picture,output_path):
    #scale factor for picture
    scale_factors = [0.5, 0.25, 0.75, 1, 1.5]
    windows.resize_images(path_imput_picture,output_path, scale_factors)
    #crop all the picture in smaller picture with img_size and overlaps factor
    windows.crop_images(path_imput_picture,output_path,(160, 240),0.5)
    
 

print("Generating data-set for training...")

if os.path.exists('dataset-premierclassifieur/images_transfomes'):
    print("Le dossier existe déjà.")
    user_input = input("Voulez-vous l'écraser ? (Oui/Non) ")
    if user_input.lower() == 'oui':
        #deletting data
        shutil.rmtree('dataset-premierclassifieur/images_transfomes')
        os.mkdir('dataset-premierclassifieur/images_transfomes')
        gen_data()
        pass
    else:
        # Ajoutez ici le code pour sortir ou effectuer une autre action
        pass
else:
    print("no history of data, generating..")
    os.mkdir('dataset-premierclassifieur/images_transfomes')
    gen_data()
    pass

print("Generating data-set for boost-training")

if os.path.exists('dataset-fenetre_glissante'):
    print("Le dossier existe déjà.")
    user_input = input("Voulez-vous l'écraser ? (Oui/Non) ")
    if user_input.lower() == 'oui':
        #deletting data
        shutil.rmtree('dataset-fenetre_glissante')
        os.makedirs('dataset-fenetre_glissante')
        windows_cropping("dataset-original/train/images/pos")
        pass
    else:
        # Ajoutez ici le code pour sortir ou effectuer une autre action
        pass
else:
    print("no history of data, generating..")
    os.makedirs('dataset-fenetre_glissante')
    windows_cropping("dataset-original/train/images/pos")
    pass


print("Generating data-set for test")

if os.path.exists('dataset-test'):
    print("Le dossier existe déjà.")
    user_input = input("Voulez-vous l'écraser ? (Oui/Non) ")
    if user_input.lower() == 'oui':
        #deletting data
        shutil.rmtree('dataset-test')
        os.makedirs('dataset-test')
        windows_cropping("dataset-original/test",'dataset-test')
        pass
    else:
        # Ajoutez ici le code pour sortir ou effectuer une autre action
        pass
else:
    print("no history of data, generating..")
    os.makedirs('dataset-test')
    windows_cropping("dataset-original/test",'dataset-test')
    pass








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

