import tkinter as tk
import tkinter.font as tkFont
from classifieurs.classifiers import *
from skimage.feature import hog
from skimage import io, util
import numpy as np
import glob
import cv2
import shutil

SIZE = 4

class DisplayImage:
    def __init__(self):
        self.m_canvas = None
        self.m_text_indicator = tk.StringVar()
        self.m_current_index = 0
        self.m_filenames = None

        self.m_save_index = 0

        self.m_text_indicator.set("Image " + str(self.m_current_index + 1) + "/" + str(len(positive_images_filenames)))
    
    def next(self):
        self.m_current_index = (self.m_current_index + 1) % len(self.m_filenames)
        self.m_text_indicator.set("Image " + str(self.m_current_index + 1) + "/" + str(len(self.m_filenames)))
        self.display()
    
    def copy_image(self):
        shutil.copyfile(self.m_filenames[self.m_current_index], "dataset-classifieur_ameliore/neg/false_positive_" + str(self.m_save_index) + ".jpg")
    
    def display(self):
        self.m_canvas.delete("all")
        I = io.imread(self.m_filenames[self.m_current_index])
        I = util.img_as_ubyte(I)
        I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
        I = cv2.resize(I, (160*SIZE, 240*SIZE))
        I = tk.Image.fromarray(I)
        I = tk.ImageTk.PhotoImage(I)
        self.m_canvas.create_image(0, 0, anchor=tk.NW, image=I)
        self.m_canvas.image = I
    
    def copy_and_next(self):
        self.copy_image()
        self.next()




X_data = []
Y_data = []

print("Importing train data...")

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

print("Train data imported.")

# Train AdaBoost classifier

print("Training AdaBoost classifier...")

classifier = AdaBoost(400)
classifier.train(X_data, Y_data)

print("AdaBoost classifier trained.")

# Import test data

print("Importing test data...")

X_test_data = []
test_data_filenames = []

for filename in glob.glob("dataset-xxx/*.jpg"):
    I = io.imread(filename)
    if (I.shape[0] == 240 and I.shape[1] == 160):
        I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
        hog_image = hog(I, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=False)
        X_test_data.append(hog_image)
        test_data_filenames.append(filename)

X_test_data = np.array(X_test_data)
test_data_filenames = np.array(test_data_filenames)

print("Test data imported.")

# Predict test data

print("Predicting test data...")

Y_test_data = classifier.predict(X_test_data)

print("Test data predicted.")

number_of_positive_images = np.count_nonzero(Y_test_data == 1)

ans = input("Positive images: " + str(number_of_positive_images) + "/" + str(len(X_test_data)) + ". Do you want to run the selector app ? (y/n) ")

positive_images_filenames = test_data_filenames[Y_test_data == 1]

if (ans == "n"):
    exit()

# Run selector app

display = DisplayImage()
display.m_filenames = positive_images_filenames

window = tk.Tk()
window.title("False positive selector")

displayer = tk.Frame(window, bg="white", bd=0)
displayer.pack(side=tk.LEFT, padx=0)

widgets = tk.Frame(window, bg="#e6e6e6", bd=0, width=400, height=600)
widgets.pack(side=tk.RIGHT, padx=0)

font = tkFont.Font(family="Poppins", size=10, weight="bold")

display.m_canvas = tk.Canvas(
    displayer, width=160*SIZE, height=240*SIZE, bg="white", bd=0, relief=tk.FLAT
)
display.m_canvas.pack(padx=0, pady=0)

# Create widgets

tk.Label(widgets, text=display.m_text_indicator, bg="#e6e6e6", fg="black", font=font).pack(padx=0, pady=0)
tk.Button(widgets, text="Positive", bg="#e6e6e6", fg="black", font=font, command=display.next).pack(padx=0, pady=0)
tk.Button(widgets, text="Negative", bg="#e6e6e6", fg="black", font=font, command=display.copy_and_next).pack(padx=0, pady=0)

display.display()

window.mainloop()
