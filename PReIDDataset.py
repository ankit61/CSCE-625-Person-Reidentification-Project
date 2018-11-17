import torch
import torch.utils.data
import os
from enum import Enum
from PIL import Image

class DatasetType(Enum):
	TRAIN = 1
	VAL = 2
	TEST = 3

class PReIDDataset(torch.utils.data.Dataset):
	def __init__(
		self, 
		load,
		transform=None, 
		trainPath = "/datasets/DukeMTMC-reID/bounding_box_train/train",
		valPath = "/datasets/DukeMTMC-reID/bounding_box_train/val",
		testPath = "/datasets/DukeMTMC-reID/bounding_box_test/"
	):
		if(load == DatasetType.TRAIN):
			self.imgfilenames = sorted([filename for _, _, filename in os.walk(trainPath)][0])
			self.path = trainPath
		elif(load == DatasetType.VAL):
			self.imgfilenames = sorted([filename for _, _, filename in os.walk(valPath)][0])
			self.path = valPath
		else:
			self.imgfilenames = sorted([filename for _, _, filename in os.walk(testPath)][0])
			self.path = testPath
		
		self.transform = transform
	
	def __len__(self):
		return len(self.imgfilenames)

	def __getitem__(self, key):

		img = Image.open(os.path.join(self.path, self.imgfilenames[key])).convert('RGB')

		if self.transform is not None:
			img = self.transform(img)
		
		#print(_img.size())

		return (img, img) #target is also the image

