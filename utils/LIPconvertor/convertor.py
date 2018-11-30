import cv2
import torch
import os

PATH_TO_LIP = "/datasets/LIP/"

FILESETS = [
    #"Testing_images/testing_images/",
    #"train_segmentations_reversed/",
    #"TrainVal_images/train_images/",
    #"TrainVal_images/val_images/",
    "EDITED_TrainVal_parsing_annotations/train_segmentations/",
    "EDITED_TrainVal_parsing_annotations/val_segmentations/"
]

for filename in FILESETS:
    path = PATH_TO_LIP + filename
    image_names = os.listdir(path) 
    for name in image_names:
        img = cv2.imread(path + name)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i,j,0] > 0:
                    #print(img[i,j])
                    img.itemset((i,j,0), 1)
                    img.itemset((i,j,1), 1)
                    img.itemset((i,j,2), 1)
        cv2.imwrite(path + name, img)
        #break

