from classifieurs.classifiers import *
from operation_sur_images.fenetre_glissantes import windows
from operation_sur_images.resize import label_image
from false_positive_detector.automatic_detector import *
from skimage.feature import hog
from skimage import io, util
import numpy as np
import glob
import cv2
import os
import shutil

X_data = []
Y_data = []

#### Generating train data

def gen_data():
    
    label_image.crop_and_save_images("dataset-original/train/labels_csv","dataset-original/train/images/pos","dataset-premierclassifieur/images_transfomes/crop/pos")
    
    label_image.crop_to_ratio('dataset-premierclassifieur/images_transfomes/crop/pos')
    
    label_image.resize_images('dataset-premierclassifieur/images_transfomes/crop/pos/cropped',"dataset-premierclassifieur/images_transfomes/images_pretes_a_etre_utliser/pos",(160,240))

    label_image.flip_images('dataset-premierclassifieur/images_transfomes/images_pretes_a_etre_utliser/pos')
    
    label_image.random_crop_images("dataset-original/train/images/neg", "dataset-premierclassifieur/images_transfomes/crop/neg")

    label_image.resize_images('dataset-premierclassifieur/images_transfomes/crop/neg','dataset-premierclassifieur/images_transfomes/images_pretes_a_etre_utliser/neg',(160,240))
       
    label_image.flip_images('dataset-premierclassifieur/images_transfomes/images_pretes_a_etre_utliser/neg')

  

    

print("Generating data-set for training...")

if os.path.exists('dataset-premierclassifieur/images_transfomes'):
    print("Le dossier existe déjà.")
    user_input = input("Voulez-vous l'écraser ? (Oui/Non) ")
    if user_input.lower() == 'oui':
        #deletting data
        shutil.rmtree('dataset-premierclassifieur/images_transfomes')
        os.makedirs('dataset-premierclassifieur/images_transfomes')
        gen_data()
        pass
    else:
        # Ajoutez ici le code pour sortir ou effectuer une autre action
        pass
else:
    print("no history of data, generating..")
    os.makedirs('dataset-premierclassifieur/images_transfomes')
    gen_data()
    pass

print("Generating data-set for boost-training")

user_input = input("Voulez-vous Generer les données ? (Oui/Non) ")
if user_input.lower() == 'oui':
    boost_training_img=windows.crop_images(windows.resize_images("dataset-original/train/images/pos",[0.5, 0.25, 0.75, 1, 1.5]),(160, 240),0.5)
    pass
else:
    pass


print("Generating data-set for test")
user_input = input("Voulez-vous Generer les données ? (Oui/Non) ")
if user_input.lower() == 'oui':
    Windows_test_img=windows.crop_images(windows.resize_images("dataset-original/test",[0.5, 0.25, 0.75, 1, 1.5,2]),(160, 240),0.75)
    pass
else:
    pass


#### Add false positive to train data

print("Training second classifier...")
print("Importing test data...")
copy_dataset_to_new_folder()

X_test_data, test_data_filenames = import_test_data()
real_rectangles = import_real_rectangles()
print("Test data imported")
print("Training classifier")
classifier = get_trained_classifier()

positive_images, scores = get_positive(classifier, X_test_data, test_data_filenames)

fp = get_false_positive_rectangles(positive_images, scores, real_rectangles)

while(len(fp)/len(positive_images) > 0.1):
    print(str(len(fp))+"/"+str(len(positive_images))+" false positive")
    print("Add false positive to folder")
    #Get 10% of false positive images
    false_positive_rectangles = fp[:max(int(len(fp)*0.1), 1)]
    copy_false_positive_images(false_positive_rectangles)

    # Restart the process until there is no false positive
    print("Training classifier")
    classifier = get_trained_classifier()
    positive_images, scores = get_positive(classifier, X_test_data, test_data_filenames)
    fp = get_false_positive_rectangles(positive_images, scores, real_rectangles)

print("Training done !")