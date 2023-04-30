import cv2
import glob
import os



os.makedirs("rectangles")

for filename in glob.glob("dataset-original/test/*.jpg"):
    csvfile = open("rectangles.csv", "r")
    I = cv2.imread(filename)
    for line in csvfile:
        line = line.split(",")
        if (int(line[0]) == int(filename.split("\\")[-1][:-4])):
            cv2.rectangle(I, (int(line[2]), int(line[1])), (int(line[2])+int(line[4]), int(line[1])+int(line[3])), (0, 255, 0), 2)
    cv2.imwrite("rectangles/"+filename.split("\\")[-1], I)
    csvfile.close()
    