import cv2
import torch
import os
import numpy as np
PATH_TO_LIP = "/datasets/LIP/TrainVal_images/train_images/"

path = PATH_TO_LIP

image_names = os.listdir(path) 

mean = [0,0,0]
std = [0,0,0]

length = len(image_names)

for n in range(0, length):
    
    temp  = cv2.meanStdDev(cv2.imread(path + image_names[n]))
    
    mean[0] += temp[0][0]
    mean[1] += temp[0][1]
    mean[2] += temp[0][2]
    std[0] += temp[1][0]
    std[1] += temp[1][1]
    std[2] += temp[1][2]

mean[0] /= length * 256.0 
mean[1] /= length * 256.0 
mean[2] /= length * 256.0 
std[0] /= length * 256.0 
std[1] /= length * 256.0 
std[2] /= length * 256.0 

print(mean)
print(std)
