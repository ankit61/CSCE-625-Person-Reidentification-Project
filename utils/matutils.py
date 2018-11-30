import os
import torch
import scipy.io
import argparse
import pandas as pd
import numpy as np

def fliplr(img):
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip

def extractor(model, dataloader):

    test_names = []
    test_features = torch.FloatTensor()

    #dataloader provides the entire batch and one image from that batch (likely picked at random until every image is used)
    for batch, sample in enumerate(tqdm(dataloader)):

        #get a name and an image
        names, images = sample['name'], sample['img']

        #run that image through the network for the 
        ff = model(Variable(images.cuda(), volatile=True))[0].data.cpu()
        ff = ff + model(Variable(fliplr(images).cuda(), volatile=True))[0].data.cpu()
        
        #normalize the 
        ff = ff.div(torch.norm(ff, p=2, dim=1, keepdim=True).expand_as(ff))

        test_names = test_names + names
        test_features = torch.cat((test_features, ff), 0)

    return test_names, test_features
"""
def calculatemAP(imagelabel, guesslist, ignorelabel=None):
    count = 1.0
    totalmAP = 0.0
    for n in range(1,len(guesslist) + 1):
        if ignorelabel == guesslist[n-1]:
            pass
        elif imagelabel == guesslist[n-1]: 
            totalmAP += count / float(n)
            count += 1

    totalmAP /= (count - 1)
    return totalmAP

    import os
import torch
import scipy.io
import argparse
import pandas as pd
import numpy as np
"""