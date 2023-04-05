import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as patches
from  scipy.interpolate  import  griddata
import numpy as np
import random
import os
from PIL import Image


def printpositif():

    labels = os.listdir(os.path.join("images/dataset-main/train", "labels_csv"))
    # Affiche au hasard des images positives de l'ensemble d'apprentissage
    fig, axs = plt.subplots(2, 5)
    for ax in axs.ravel():
        k = random.randrange(0, len(labels))
        img = plt.imread(os.path.join("images/dataset-main/train", "images", "pos", labels[k][:-4] + ".jpg"))
        bbox = np.loadtxt(os.path.join("images/dataset-main/train", "labels_csv", labels[k]), delimiter=",")
        if bbox.ndim == 1:
            bbox = [bbox]

            ax.axis('off')
            ax.set_title(labels[k][:-4])
            ax.imshow(img)
        for bb in bbox:
            rect = patches.Rectangle((bb[1], bb[0]), bb[3], bb[2], linewidth=1, 
                                 edgecolor='r' if bb[4] > 0 else 'g', facecolor='none')
            ax.add_patch(rect)
    plt.show()


#printpositif()


def rename_files(path,name):
    i = 1
    for filename in os.listdir(path):
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png') or filename.endswith('.csv'):
            new_filename = '{:03d}.{}'.format(i, filename.split('.')[-1])
            os.rename(os.path.join(path, filename), os.path.join(path, new_filename))
            i += 1


def crop_and_save_images():
    
    label_folder='dataset-premierclassifieur/labels_csv'
    #img_foldernegative='dataset-premierclassifieur/images/neg'
    img_folderpostive= 'dataset-premierclassifieur/images/pos'
    output_folder='dataset-premierclassifieur/images_transfomes/crop/pos'
    output_folder2='dataset-premierclassifieur/images_transfomes/crop/neg'
    labels = os.listdir(label_folder)
    facteur=1.1
    for label in labels:
        # Charger l'image
        img_path = os.path.join(img_folderpostive, label[:-4] + ".jpg")
        img = Image.open(img_path)
        #img2 = Image.open(img_foldernegative)
        # Charger les bboxes depuis le fichier CSV
        bbox = np.loadtxt(os.path.join(label_folder, label), delimiter=",")
        if bbox.ndim == 1:
            bbox = [bbox]

        # Extraire et enregistrer chaque crop
        for bb in bbox:
            #on prend que les ecocup "facile" et que celle avec un ratio entre 1.5 et 2
            if bb[4]==0 and (1.5<=bb[2]/bb[3]<=2):
                crop = img.crop(((bb[1])*(0.9), (bb[0])*(0.9), (bb[1])*facteur+(bb[3])*facteur, (bb[0])*facteur+(bb[2])*facteur)) # (left, upper, right, lower)
                #crop2=img2.crop(((bb[1])*(0.9), (bb[0])*(0.9), (bb[1])*facteur+(bb[3])*facteur, (bb[0])*facteur+(bb[2])*facteur)) # (left, upper, right, lower)
                # Redimensionner le crop
                crop_resized = np.array(crop)#.resize(output_size))
                #crop_resized2=np.array(crop2)
                # Enregistrer le crop dans un fichier
                filename = label[:-4] + f"_crop_{bb[1]}_{bb[0]}_{bb[3]}_{bb[2]}.jpg"
                filepath = os.path.join(output_folder, filename)
                #filepath2= os.path.join(output_folder2, filename)
                plt.imsave(filepath, crop_resized)
                #plt.imsave(filepath2, crop_resized2)
                # Afficher le crop
                # fig, ax = plt.subplots(1)
                # ax.imshow(crop_resized, extent=(bb[1], bb[1]+bb[3], bb[0]+bb[2], bb[0]))
                # ax.axis('off')
                # ax.set_title(filename)
                # plt.show()

#crop_and_save_images()

def random_crop_images(input_folder, output_folder):
    # Vérifiez si le dossier de sortie existe, sinon, créez-le.
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Boucle à travers chaque fichier dans le dossier d'entrée.
    for filename in os.listdir(input_folder):
        # Vérifiez si le fichier est une image.
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            # Ouvrez le fichier image en utilisant Pillow.
            image = Image.open(os.path.join(input_folder, filename))

            # Obtenez la largeur et la hauteur actuelles de l'image.
            width, height = image.size

            # Calculez un ratio de crop aléatoire entre 1.5 et 2.
            ratio = 1.5

            # Calculez une taille de crop aléatoire entre 25% et 50% de la taille de l'image originale.
            crop_size = int(np.random.uniform(low=0.25, high=0.5) * min(width, height))

            # Calculez les nouvelles dimensions de l'image en utilisant le ratio et la taille de crop aléatoires.
            new_width = crop_size
            new_height = int(ratio * crop_size)

            # Générez une position de crop aléatoire.
            x = np.random.randint(0, width - new_width)
            y = np.random.randint(0, height - new_height)

            # Récupérez le crop à partir de l'image originale.
            crop = image.crop((x, y, x + new_width, y + new_height))

            # Enregistrez le crop dans le dossier de sortie.
            crop.save(os.path.join(output_folder, filename[:-4] + '_' + str(ratio)[:3] + '_' + str(crop_size) + '.jpg'))

#random_crop_images("dataset-premierclassifieur/images/neg", "dataset-premierclassifieur/images_transfomes/crop/neg")


def resize_images(folder_path, size):
    """
    Redimensionne toutes les images présentes dans le dossier spécifié par
    folder_path en utilisant l'interpolation linéaire, et enregistre les
    images redimensionnées dans un nouveau dossier.
    """
    # Créer un nouveau dossier pour les images redimensionnées.
    new_folder_path = os.path.join(folder_path, "resized")
    os.makedirs(new_folder_path, exist_ok=True)

    # Obtenir la liste de tous les fichiers dans le dossier.
    files = os.listdir(folder_path)

    # Parcourir la liste de fichiers et redimensionner chaque image.
    for file in files:
        # Vérifier si le fichier est une image (extension jpg, png, etc.).
        if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
            # Ouvrir l'image.
            image_path = os.path.join(folder_path, file)
            image = Image.open(image_path)

            # Redimensionner l'image.
            resized_image = image.resize(size, Image.LINEAR)

            # Enregistrer la nouvelle image.
            new_image_path = os.path.join(new_folder_path, file)
            resized_image.save(new_image_path)

#resize_images('dataset-premierclassifieur/images_transfomes/crop/neg','dataset-premierclassifieur/images_transfomes/resize/neg',(160,240))
#resize_images('dataset-premierclassifieur/images_transfomes/images_pretes_a_etre_utliser/pos',(160,240))

def flip_images(folder_path):
    """
    Effectue une symétrie verticale pour toutes les images présentes
    dans le dossier spécifié par folder_path.
    """
    # Obtenir la liste de tous les fichiers dans le dossier.
    files = os.listdir(folder_path)

    # Parcourir la liste de fichiers et effectuer la symétrie pour chaque image.
    for file in files:
        # Vérifier si le fichier est une image (extension jpg, png, etc.).
        if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
            # Ouvrir l'image.
            image_path = os.path.join(folder_path, file)
            image = Image.open(image_path)

            # Effectuer la symétrie.
            flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)

            # Enregistrer la nouvelle image.
            new_image_path = os.path.join(folder_path, "flipped_" + file)
            flipped_image.save(new_image_path)

#flip_images('dataset-premierclassifieur/images_transfomes/resize/neg')
#flip_images('dataset-premierclassifieur/images_transfomes/resize/pos')


def crop_to_ratio(folder_path):
    """
    Redimensionne toutes les images dans le dossier spécifié en les recadrant
    pour obtenir un ratio hauteur/largeur de 1.5.

    Args:
        folder_path (str): Chemin vers le dossier contenant les images.
    """
    # Créer un dossier pour enregistrer les images recadrées.
    cropped_folder_path = os.path.join(folder_path, "cropped")
    os.makedirs(cropped_folder_path, exist_ok=True)

    # Obtenir la liste de tous les fichiers dans le dossier.
    files = os.listdir(folder_path)

    # Parcourir la liste de fichiers et effectuer le recadrage pour chaque image.
    for file in files:
        # Vérifier si le fichier est une image (extension jpg, png, etc.).
        if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
            # Ouvrir l'image.
            image_path = os.path.join(folder_path, file)
            image = Image.open(image_path)

            # Récupérer les dimensions de l'image.
            width, height = image.size

            # Calculer la hauteur cible en fonction de la largeur.
            target_height = int(width * 1.5)

            # Si l'image est plus haute que la cible, couper des parties des bords supérieurs et inférieurs.
            if height > target_height:
                # Calculer la hauteur à couper.
                crop_height = int((height - target_height) / 2)

                # Effectuer le crop.
                image = image.crop((0, crop_height, width, height - crop_height))

            # Si l'image est plus large que la cible, couper des parties des bords gauche et droit.
            elif width > int(height / 1.5):
                # Calculer la largeur à couper.
                target_width = int(height / 1.5)
                crop_width = int((width - target_width) / 2)

                # Effectuer le crop.
                image = image.crop((crop_width, 0, width - crop_width, height))

            # Redimensionner l'image pour qu'elle ait exactement la taille cible.
            image = image.resize((int(target_height / 1.5), target_height))

            # Enregistrer la nouvelle image.
            cropped_image_path = os.path.join(cropped_folder_path, file)
            image.save(cropped_image_path)
            
#crop_to_ratio('dataset-premierclassifieur/images_transfomes/crop/pos')
