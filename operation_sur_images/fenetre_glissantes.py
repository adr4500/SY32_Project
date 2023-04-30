import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as patches
from  scipy.interpolate  import  griddata
import numpy as np
import random
import os
from PIL import Image
import cv2
from pathlib import Path

class windows:
    
    def show_same_prefix_images(folder_path, prefix):
        """
        Affiche toutes les images du dossier spécifié par folder_path
        qui ont le même préfixe que celui spécifié par prefix,
        en les affichant dans le même canvas.
        """
        # Obtenir la liste de tous les fichiers dans le dossier.
        files = os.listdir(folder_path)

        # Parcourir la liste de fichiers et afficher les images avec le même préfixe.
        images = []
        for file in files:
            # Vérifier si le fichier est une image (extension jpg, png, etc.)
            if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
                # Vérifier si le préfixe est présent au début du nom de fichier.
                if file.startswith(prefix + '_'):
                    # Ouvrir l'image et l'ajouter à la liste des images.
                    image_path = os.path.join(folder_path, file)
                    image = Image.open(image_path)
                    images.append(image)

        # Calculer la taille totale du canvas.
        canvas_width = sum([image.width for image in images])
        canvas_height = max([image.height for image in images])

        # Créer le canvas et copier les images dedans.
        canvas = Image.new("RGB", (canvas_width, canvas_height))
        x = 0
        for image in images:
            canvas.paste(image, (x, 0))
            x += image.width

        # Afficher le canvas.
        canvas.show()
    #show_same_prefix_images("dataset-original/train/images/dossier_de_test/output",'abigotte')
    
    def resize_images(img_folder, scale_factors):
        

        # Lister les fichiers dans le dossier d'entrée
        file_list = os.listdir(img_folder)
        resized_images = []
        
        for filename in file_list:
            # Lire l'image d'origine
            img = cv2.imread(os.path.join(img_folder, filename))

            # Appliquer chaque facteur d'échelle
            for scale_factor in scale_factors:
                # Redimensionner l'image en conservant les proportions
                img_resized = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
                #num, denom = scale_factor.as_integer_ratio()
                # Enregistrer l'image redimensionnée dans le dossier de sortie
                output_filename = f"{os.path.splitext(filename)[0]}-{float(scale_factor)}.jpg"
                
                resized_images.append((output_filename, img_resized))

        return resized_images
            
    

    def crop_images(resized_files,crop_size, overlap):
        """
        Découpe toutes les images du dossier spécifié par input_path en plusieurs images
        de taille crop_size, avec un chevauchement de overlap, et enregistre ces images
        dans le dossier spécifié par output_path.
        """
        cropped_img=[]
        # Parcourt tous les fichiers du dossier input_path.
        for filename, img in resized_files:
        # Obtient les dimensions de l'image.
            # Vérifie que le fichier est une image.
            #on recup le facteur d'echelle de l'image pour faire la conversion vers image originale
            img_name_without_ext = os.path.splitext(os.path.basename(filename))[0]
            #je recup le nominateur et denom  du facteur d'echelle dans le nom du fichier de l'image
            indice= img_name_without_ext.split("-")[-1]
            indice=float(indice)
            # Obtient les dimensions de l'image.
            width, height = img.shape[1], img.shape[0]

            # Initialise les coordonnées de départ pour le crop.
            y_start = 0

            # Parcourt toutes les lignes de l'image.
            while y_start < height:
                x_start = 0
                # Parcourt toutes les colonnes de l'image.
                while x_start < width:
                    # Calcule les coordonnées de fin du crop.
                    x_end = min(x_start + crop_size[0], width)
                    y_end = min(y_start + crop_size[1], height)

                    # Effectue le crop.
                    cropped_image = img[int(y_start):int(y_end), int(x_start):int(x_end), :]

                    # Construit le nom de l'image enregistrée.
                    i = y_start/indice
                    j = x_start/indice
                    name = f"{filename.split('-')[0]}-{i}-{j}-{crop_size[1]/indice}-{crop_size[0]/indice}.jpg"
                    #Enregistre l'image cropée.
                    #print(cropped_image.size[0])
                    if (cropped_image.shape[0]==240 and cropped_image.shape[1]== 160)or(cropped_image.shape[0]==160 and cropped_image.shape[1]== 240):
                        cropped_img.append((name, cropped_image))

                    # Passe à la prochaine colonne.
                    x_start += crop_size[0] - (overlap*crop_size[0])

                    # Passe à la prochaine ligne.
                y_start += crop_size[1] - (overlap*crop_size[1])

        return cropped_img
    
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
            
            
            
            
            
            
            
            
            
