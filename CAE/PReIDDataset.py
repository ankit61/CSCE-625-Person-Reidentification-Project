import torch
import torch.utils.data
import os
from enum import Enum
from PIL import Image

class PReIDDataset(torch.utils.data.Dataset):
	def __init__(
		self, 
		path,
		transform=None, 
	):
		self.imgfilenames = sorted([filename for _, _, filename in os.walk(path)][0])
		self.path = path
		self.transform = transform
	
	def __len__(self):
		return len(self.imgfilenames)

	def __getitem__(self, key):

		img = Image.open(os.path.join(self.path, self.imgfilenames[key])).convert('RGB')
		personID = int(self.imgfilenames[key][0:4])

		if self.transform is not None:
			img = self.transform(img)
		
		#print(_img.size())

		return (img, img, personID, self.imgfilenames[key]) #target is also the image

	def getFileName(self, i):
		return self.imgfilenames[i]

