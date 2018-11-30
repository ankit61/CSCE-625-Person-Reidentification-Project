import os
import torch
import scipy.io
import argparse
import numpy as np


from generateClusters import generateResults

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


