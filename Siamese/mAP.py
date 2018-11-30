#!/usr/bin/env python3
import os
import torch
import scipy.io
import argparse
import numpy as np
from dataset import SimpleDataset
import torchvision.transforms as transforms
import torch.nn.functional as F
from network import Siamese
from PIL import Image
import sys

img_size = (208, 76)
mean = [0.216, 0.2074816, 0.22934238]
std = [0.2333638, 0.22653223, 0.23671082]

#############################################################################
# Computing mAP
#############################################################################
#THIS NEEDS TO BE LOOKED AT AGAIN!!!!!!
def compute_mAP(index, good_index, junk_index, bad_index):
	ap = 0
	cmc = torch.IntTensor(len(index)).zero_()
	if good_index.size == 0:
		cmc[0] = -1
		return ap, cmc

	# remove junk_index
	mask = np.in1d(index, junk_index, invert=True)
	index = index[mask]
	#print(index)

	#find good_index index
	ngood = len(good_index)
	mask = np.in1d(index, good_index)
	#print(mask)

	rows_good = np.argwhere(mask == True)
	rows_good = rows_good.flatten()
	
	#Part that Benton wrote because Ye's stuff was throwing an error
	#for i in range(0, len(bad_index)):
	#	index[bad_index[i]] = 0 
	#rows_good = index
	
	
	#Part where mAP actually gets calculated

	cmc[rows_good[0]:] = 1
	for i in range(ngood):
		d_recall = 1.0 / ngood
		precision = (i + 1) * 1.0 / (rows_good[i] + 1)
		if rows_good[i] != 0:
			old_precision = i * 1.0 / rows_good[i]
		else:
			old_precision = 1.0
		ap = ap + d_recall * (old_precision + precision) / 2

	return ap, cmc[:20]


def evaluate(ql, qc, gl, gc):
	# good index
	query_index = np.argwhere(gl == ql)
	camera_index = np.argwhere(gc == qc)

	good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
	
	#hotfix for the code to work without cameras
	for i in range(0,len(good_index)):
		good_index[i] = gl[good_index[i]]  	
	#print(good_index)

	junk_index1 = np.argwhere(gl == -1)
	junk_index2 = np.intersect1d(query_index, camera_index)
	junk_index = np.append(junk_index2, junk_index1)

	bad_index = np.argwhere(gl != ql)
	bad_index = bad_index.flatten(-1)
	
	return compute_mAP(gl, good_index, junk_index, bad_index)

def createStats(query, gallery):
	imgclass = int(query[0:4])
	labels = [int(label[0:4]) for label in gallery]
	score_tup = evaluate(np.array([imgclass]), [], np.array(labels), [])
	return score_tup


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
				#dists.append((F.mse_loss(gallery_feature, query_feature), filename))
				dists.append((gallery_feature, filename))

	#   img = utils.make_grid(torch.cat((input_img, output), 0), nrow=2)
		return dists

def generateResults(query_path, gallery_path, model=None, modelpath="/home/ankit/csce-625-person-re-identification/Siamese/trained_resnets/checkpoint_33.tar"):
		if model:
			model.cuda()
		else:
			checkpoint = torch.load(model_path)
			model = Siamese()
			model.cuda()
			model.load(checkpoint['state_dict'])

		imgTransforms = transforms.Compose([
				transforms.Resize(img_size),
				transforms.ToTensor(),
				transforms.Normalize(mean=mean, std=std),
		])
		query_list = sorted([filename for _, _, filename in os.walk(query_path)][0])
		count = 1
		total = len(query_list)
		CMC = 0
		ap = 0

		raw_dists = [] #temporary

		for img in query_list:
			dists = [] #temporary

			if count % 10 == 1: 
				print(f"starting {count}/{total}")
			
			query_tensor = imgTransforms(Image.open(os.path.join(query_path, img)).convert('RGB'))
			query_tensor = torch.autograd.Variable(query_tensor, volatile=True).cuda()
			query_tensor = query_tensor.view(1, query_tensor.size(0), query_tensor.size(1), query_tensor.size(2))

			gallery_loader = torch.utils.data.DataLoader(
							SimpleDataset(path=gallery_path, transforms=imgTransforms),
							batch_size=1, shuffle=False,
							num_workers=4, pin_memory=True)
			
			### TEMPORARY SOLUTION TO THE GALLERY IMAGE PROBLEM ###
			if count == 1:
				raw_dists = getDists(query_tensor, gallery_loader, model)

			# compute output
			model.eval()
			query_feature, _ = model(query_tensor, query_tensor)

			
			for dist in raw_dists:
				dists.append((F.mse_loss(dist[0], query_feature), dist[1]))

			#######################################################

			dists.sort(key = lambda val : val[0])

			bestImgNames = [el[1][0] for el in dists]

			#print(bestImgNames)

			ap_tmp, CMC_tmp = createStats(img, bestImgNames) 
			if CMC_tmp[0] == -1:
				continue

			CMC += CMC_tmp
			ap += ap_tmp
			count += 1
		CMC = CMC.float()
		CMC /= len(query_list)
		ap /= len(query_list)

		#print('top1: %.4f, top5: %.4f, top10: %.4f, mAP: %.4f' % (CMC[0], CMC[4], CMC[9], ap))
		return (CMC[0], CMC[4], CMC[9], ap)


def main_test():

	query_path  = sys.argv[1]
	gallery_dir = sys.argv[2]
	
	gallery_imgfiles = np.array([filename for _, _, filename in os.walk(gallery_dir)][0])
	print(len(gallery_imgfiles))

	print(createStats(query_path, gallery_imgfiles))

	generateResults("/datasets/TAMUvalSegmented/query/", "/datasets/TAMUvalSegmented/gallery/", model_path="/home/ankit/csce-625-person-re-identification/Siamese/trained_resnets/checkpoint_33.tar")

if __name__=="__main__":
	main_test()

