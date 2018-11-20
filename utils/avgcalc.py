#!/usr/bin/env python3
import cv2
import torch
import os
import numpy as np
import sys

path = sys.argv[1]

image_names = os.listdir(path) 

mean = [0,0,0]
std = [0,0,0]
size = [0,0];

length = len(image_names)

print("Output is displayed in BGR fashion, not RGB like pytorch expects")

for n in range(0, length):
	im = cv2.imread(os.path.join(path + image_names[n]))
	if(im is None):
		raise Exception("incorrect path: " + os.path.join(path + image_names[n]))
	temp  = cv2.meanStdDev(im)	  
	
	mean[0] += temp[0][0]
	mean[1] += temp[0][1]
	mean[2] += temp[0][2]
	
	std[0] += temp[1][0]
	std[1] += temp[1][1]
	std[2] += temp[1][2]

	size[0] += im.shape[0] 
	size[1] += im.shape[1]

mean[0] /= length * 255.0 
mean[1] /= length * 255.0 
mean[2] /= length * 255.0 

std[0] /= length * 255.0 
std[1] /= length * 255.0 
std[2] /= length * 255.0 

size[0] /= length
size[1] /= length

print("mean:", mean)
print("std:", std)
print("size:", size)
