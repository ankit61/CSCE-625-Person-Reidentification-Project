import sys
import os
import torch
import torchvision.transforms as transforms
import PIL
from PIL import Image
from itertools import combinations, product
from functools import reduce
from random import shuffle, sample

def random_combination(iterable, r):
    "Random selection from itertools.combinations(iterable, r)"
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(sample(range(n), r))
    return [pool[i] for i in indices]

class SiameseSampler(torch.utils.data.Sampler):
    def __init__(self, data_source):
        self.classes = data_source.getList()
        # construct the keys
        self.total = []
        diff_pairs = random_combination(combinations(self.classes, 2), 2250)
        same_pairs = sample([(c, c) for c in self.classes], 250)

        #print(len(diff_pairs))
        #print(len(same_pairs))
        pairs = diff_pairs + same_pairs
        for pair in pairs:
            rng1 = range(0, data_source.getClassLength(pair[0]))
            rng2 = range(0, data_source.getClassLength(pair[1]))
            rngboth = tuple(product(rng1, rng2)) 
            sample_n = 5
            if len(rngboth) < 5:
                sample_n = len(rngboth)
            indices = random_combination(rngboth, sample_n)

            for tup in indices:
                self.total.append(
                    (pair[0], pair[1], tup[0], tup[1])
                )

        shuffle(self.total)

    def __len__(self):
        return len(self.total)
    def __iter__(self):
        return iter(self.total)

class SiameseDataset(torch.utils.data.Dataset):
	def __init__(self, path, transforms, test=False, valpath=None):
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
		
		self.transforms = transforms
	
	def getClassLength(self, classname):
		return len(self.dataclass[classname])
	
	def getList(self):
		return [name for name in self.dataclass]
	
	def __len__(self):
		return len(self.dataclass)
	
	def __getitem__(self, key):
		# key structure (class1, class2, index1, index2) #
	
		imgname1 = self.dataclass[key[0]][key[2]]
		imgname2 = self.dataclass[key[1]][key[3]]

		img1 = Image.open(self.path + imgname1)
		img2 = Image.open(self.path + imgname2)
	
		if key[0] == key[1]:
			same = 1
		else:
			same = 0

		imgtensor1 = self.transforms(img1)
		imgtensor2 = self.transforms(img2)

		if test == True:
			return imgtensor1, imgtensor2, same, key[0], key[1]
		else:
			return imgtensor1, imgtensor2, same


#temporary tests
#s = SiameseDataset("/datasets/DukeSegmented/train/")

"""
print(
    s[1495, 1492, 0, 0]
)
"""

#print(s.getList())
#print(s.getClassLength(1492))
#print(s.getClassLength(1495))

#s_sampler = SiameseSampler(s)
#print (len(s_sampler))
#print ([key for key in s_sampler])
#print (s[13, 13, 33, 0])
 
