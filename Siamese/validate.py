import os
import torch
import scipy.io
import argparse
import numpy as np

writer = SummaryWriter("/runs/")

img_size = (208, 76)
mean = [0.216, 0.2074816, 0.22934238]
std = [0.2333638, 0.22653223, 0.23671082]

def getDists(query_embedding, gallery_loader, model):
	"""
	Run evaluation
	"""
	# switch to evaluate mode
	model.eval()
	dists = []
	with torch.no_grad():
		for i, (input_img, target, ID, filename) in enumerate(gallery_loader):
			target = target.cuda(async=True)
			input_var = torch.autograd.Variable(input_img, volatile=True).cuda()
			target_var = torch.autograd.Variable(target, volatile=True)

			# compute output
			model(input_var)

			#dists.append((F.mse_loss(model.embedding, query_embedding), filename))
			dists.append((F.mse_loss(model.embedding, query_embedding), filename))
						
#			img = utils.make_grid(torch.cat((input_img, output), 0), nrow=2)

	return dists

def generateResults(query_path, gallery_path, model_path, k = 5):
	checkpoint = torch.load(model_path)
	model = CAE(img_size, should_decode=False)
	model.cuda()
	model.load(checkpoint['state_dict'])

	imgTransforms = transforms.Compose([
				transforms.Resize(img_size),
				transforms.ToTensor(),
				transforms.Normalize(mean=mean, std=std),
			])
	
	query_tensor = imgTransforms(Image.open(query_path).convert('RGB'))
	query_tensor = torch.autograd.Variable(query_tensor, volatile=True).cuda()

	model(query_tensor.view(1, query_tensor.size(0), query_tensor.size(1), query_tensor.size(2)))
	query_embedding = model.embedding

	gallery_loader = torch.utils.data.DataLoader(
			PReIDDataset(path=gallery_path, transform=imgTransforms),
			batch_size=1, shuffle=False,
			num_workers=4, pin_memory=True)

	dists = getDists(query_embedding, gallery_loader, model)
	dists.sort(key = lambda val : val[0])

	bestImgNames = [ el[1] for el in dists[0:k]]

	#print(bestImgNames)
	return bestImgNames

	#img = utils.make_grid(torch.cat((input_img, output), 0), nrow=k)

#############################################################################
# Computing mAP
#############################################################################
def compute_mAP(index, good_index, junk_index):
	ap = 0
	cmc = torch.IntTensor(len(index)).zero_()
	if good_index.size == 0:
		cmc[0] = -1
		return ap, cmc

	# remove junk_index
	mask = np.in1d(index, junk_index, invert=True)
	index = index[mask]

	# find good_index index
	ngood = len(good_index)
	mask = np.in1d(index, good_index)
	rows_good = np.argwhere(mask == True)
	rows_good = rows_good.flatten()

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

	print(good_index)

	junk_index1 = np.argwhere(gl == -1)
	junk_index2 = np.intersect1d(query_index, camera_index)
	junk_index = np.append(junk_index2, junk_index1)

	return compute_mAP(gl, good_index, junk_index)

def main_test():
	imgname = "0733_c3s2_063753_02.jpg"
	imgclass = int(imgname[0:4])

	suggestion = generateResults(
		"/datasets/Market-1501Segmented/train/" + imgname, 
		"/datasets/Market-1501Segmented/train/", 
		"/home/ankit/csce-625-person-re-identification/CAE/trained_resnets/checkpoint_988.tar", 
		1000
	)[1:]

	labels = [int(label[0][0:4]) for label in suggestion]

	#print(labels)

	score_tup = evaluate(np.array([imgclass]), [], np.array(labels), [])

	print(score_tup)

