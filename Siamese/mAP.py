import os
import torch
import scipy.io
import argparse
import numpy as np
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
	print(index)

	#find good_index index
	ngood = len(good_index)
	mask = np.in1d(index, good_index)
	print(mask)

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
	print(good_index)

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

def main_test():
	query = "1234c"
	gallery = np.array([
		"1234c",
		"1234c",
		"1234c",
		"1234c",
		"3234c",
		"9234c",
		"9234c",
		"8234c",
		"3234c",
		"4234c",
		"4234c",
		"2234c",
		"2234c",
		"3234c",
		"3234c",
		"1234c",
		"3234c",
		"0234c",
		"1234c",
		"0234c",
		"1234c"
	])

	print(createStats(query, gallery))

if __name__=="__main__":
	main_test()

