import sys
import os
sys.path.append('../')
sys.path.append("/pytorch-segmentation/pytorch-segmentation-detection/")
sys.path.insert(0, '/pytorch-segmentation/pytorch-segmentation-detection/vision/')
import torch
import torchvision.transforms as transforms
import PIL
from PIL import Image
from itertools import combinations, product
from functools import reduce

from pytorch_segmentation_detection.transforms import (
    ComposeJoint,
    RandomHorizontalFlipJoint
)


class SiameseSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, start, end):
        self.classes = data_source.getList()[start:end]
        # construct the keys
        self.total = []
        pairs = combinations(self.classes, 2)
        for pair in pairs:
            rng1 = range(0, data_source.getClassLength(pair[0]))
            rng2 = range(0, data_source.getClassLength(pair[1]))
            rngboth = product(rng1, rng2)
            for tup in rngboth:
                self.total.append(
                    (pair[0], pair[1], tup[0], tup[1])
                )

    def __len__(self):
        return len(self.total)
    def __iter__(self):
        return iter(self.total)

class SiameseLoader(torch.utils.data.Dataset):
    def __init__(self, path="/datasets/DukeSegmented/train/"):
        self.path = path
        self.imgfilenames = sorted(
            [filename for _, _, filename in os.walk(path)][0])
        self.dataclass = {}
        for name in self.imgfilenames:
            classname = int(name[0:4])
            if classname not in self.dataclass:
                self.dataclass[classname] = [name]
            else:
                self.dataclass[classname].append(name)
    
    def getClassLength(self, classname):
        return len(self.dataclass[classname])
    
    def getList(self):
        return [name for name in self.dataclass]
    
    def __len__(self):
        return len(self.dataclass)
    
    def __getitem__(self, key):
        # key structure (class1, class2, index1, index2) #
        tensor_trans = ComposeJoint(
            [
                [transforms.ToTensor(), transforms.ToTensor()]
            ]
        )

        imgname1 = self.dataclass[key[0]][key[2]]
        imgname2 = self.dataclass[key[1]][key[3]]

        img1 = Image.open(self.path + imgname1)
        img2 = Image.open(self.path + imgname2)
        

        return tensor_trans([img1, img2])

#temporary tests
#s = SiameseLoader("/datasets/DukeSegmented/train/")

"""
print(
    s[1495, 1492, 0, 0]
)
"""

#print(s.getList())
#print(s.getClassLength(1492))
#print(s.getClassLength(1495))

#s_sampler = SiameseSampler(s, 0, 30)
#print (len(s_sampler))
#for key in s_sampler:
#    print(key) 