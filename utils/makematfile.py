import sys
sys.path.append("../Siamese")
import os
import torch
import scipy.io
import argparse
import pandas as pd
import numpy as np
from dataset import SimpleDataset
from network import Siamese

def fliplr(img):

	inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # generates liat from [0, H]
	img_flip = img.index_select(3, inv_idx)

	return img_flip

def extractor(_model, _simple_dataloader):

	test_names = []
	test_features = torch.FloatTensor()

	#dataloader provides the entire batch and one image from that batch (likely picked at random until every image is used)
	for images, names in enumerate(_simple_dataloader):

		#get a name and an image
		#names, images = sample['name'], sample['img']

		#run that image through the network for the 
		ff = model.forwardforone(Variable(images.cuda(), volatile=True))[0].data.cpu()
		ff = ff + model.forwardforone(Variable(fliplr(images).cuda(), volatile=True))[0].data.cpu()
		
		#normalize the 
		ff = ff.div(torch.norm(ff, p=2, dim=1, keepdim=True).expand_as(ff))

		test_names = test_names + names
		test_features = torch.cat((test_features, ff), 0)

	return test_names, test_features

if __name__=="__main__":
	model_path = "/home/ryan/csce-625-person-re-identification/Siamese/trained_resnets/bestMAP.tar"
	wang_folders = [
		[
			"/datasets/TAMUvalSegmented/query/", "/datasets/TAMUvalSegmented/gallery/"
		],
		[
			"/datasets/TAMUtestSegmented/query/", "/datasets/TAMUtestSegmented/gallery/"
		]
	]

	checkpoint = torch.load(model_path)
	model = Siamese()
	model.cuda()
	model.load(checkpoint['state_dict'])
	imgTransforms = transforms.Compose([
		transforms.Resize(img_size),
		transforms.ToTensor(),
		transforms.Normalize(mean=mean, std=std),
	])

	for dataset in wang_folders:
		for directory in dataset:

			gallery_loader = torch.utils.data.DataLoader(
				SimpleDataset(path=directory, transforms=imgTransforms),
				batch_size=64, shuffle=False,
				num_workers=4, pin_memory=True
			)

			test_names, test_features = extractor(model, )
			results = {'names': test_names, 'features': test_features.numpy()}
			scipy.io.savemat(os.path.join(log_dir, 'feature_%s_%s.mat' % (dataset, subset)), results)