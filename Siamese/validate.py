#!/usr/bin/env python3
import os
from PIL import Image
import torch
import sys
import argparse
import numpy as np
from dataset import SimpleDataset
import torchvision.transforms as transforms
import torch.nn.functional as F
from network import Siamese

img_size = (208, 76)
mean = [0.216, 0.2074816, 0.22934238]
std = [0.2333638, 0.22653223, 0.23671082]

def getDists(query_img, gallery_loader, model):
	"""
	Run evaluation
	"""
	# switch to evaluate mode
	model.eval()
	dists = []
	with torch.no_grad():
		for (gallery_img, filename) in gallery_loader:
			input_var = torch.autograd.Variable(gallery_img, volatile=True).cuda()

			# compute output
			query_feature, gallery_feature = model(query_img, input_var)
			
			##################### IS MSE RIGHT? ###############################
			dists.append((F.mse_loss(gallery_feature, query_feature), filename))
						
#			img = utils.make_grid(torch.cat((input_img, output), 0), nrow=2)

	return dists

def generateResults(query_path, gallery_path, model_path, k = 5):
	checkpoint = torch.load(model_path)
	model = Siamese() 
	model.cuda()
	model.load(checkpoint['state_dict'])

	imgTransforms = transforms.Compose([
				transforms.Resize(img_size),
				transforms.ToTensor(),
				transforms.Normalize(mean=mean, std=std),
			])
	
	query_tensor = imgTransforms(Image.open(query_path).convert('RGB'))
	query_tensor = torch.autograd.Variable(query_tensor, volatile=True).cuda()
	query_tensor = query_tensor.view(1, query_tensor.size(0), query_tensor.size(1), query_tensor.size(2))
	
	gallery_loader = torch.utils.data.DataLoader(
			SimpleDataset(path=gallery_path, transforms=imgTransforms),
			batch_size=1, shuffle=False,
			num_workers=4, pin_memory=True)

	dists = getDists(query_tensor, gallery_loader, model)
	dists.sort(key = lambda val : val[0])

	bestImgNames = [el[1] for el in dists[0:k]]

	return bestImgNames

	#img = utils.make_grid(torch.cat((input_img, output), 0), nrow=k)

def main():
	bestImgNames = generateResults(sys.argv[1], sys.argv[2], sys.argv[3])
	print(bestImgNames)

if __name__ == '__main__':
	main()
