import sys, os
# update with your path
# All the jupyter notebooks in the repository already have this
sys.path.append("/pytorch-segmentation/pytorch-segmentation-detection/")
sys.path.insert(0, '/pytorch-segmentation/pytorch-segmentation-detection/vision/')

import pytorch_segmentation_detection.models.fcn as fcns
import pytorch_segmentation_detection.models.resnet_dilated as resnet_dilated
from pytorch_segmentation_detection.transforms import (ComposeJoint, RandomHorizontalFlipJoint, RandomScaleJoint, CropOrPad, ResizeAspectRatioPreserve, RandomCropJoint)


import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms

import numbers
import random

from matplotlib import pyplot as plt

import numpy as np
from PIL import Image

from sklearn.metrics import confusion_matrix

def flatten_logits(logits, number_of_classes):
    """Flattens the logits batch except for the logits dimension"""
    
    logits_permuted = logits.permute(0, 2, 3, 1)
    logits_permuted_cont = logits_permuted.contiguous()
    logits_flatten = logits_permuted_cont.view(-1, number_of_classes)
    
    return logits_flatten

def flatten_annotations(annotations):
    
    return annotations.view(-1)

def get_valid_annotations_index(flatten_annotations, mask_out_value=255):
    
    return torch.squeeze( torch.nonzero((flatten_annotations != mask_out_value )), 1)


train_transform = ComposeJoint(
    [
        RandomHorizontalFlipJoint(),
        RandomCropJoint(crop_size=(224, 224)),
        #[ResizeAspectRatioPreserve(greater_side_size=384),
        #ResizeAspectRatioPreserve(greater_side_size=384, interpolation=Image.NEAREST)],
        
        #RandomCropJoint(size=(274, 274))
        # RandomScaleJoint(low=0.9, high=1.1),
        
        #[CropOrPad(output_size=(288, 288)), CropOrPad(output_size=(288, 288), fill=0)],
        [transforms.ToTensor(), None],
        [transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), None],
        [None, transforms.Lambda(lambda x: torch.from_numpy(np.asarray(x)).long()) ]
    ]
)


class LIPDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        transform_rule=None, 
        trainpath = "/datasets/LIP/TrainVal_images/train_images/", 
        targetpath = "/datasets/LIP/EDITED_TrainVal_parsing_annotations/train_segmentations/"
    ):
        self.trainpath = trainpath
        self.targetpath = targetpath
        self.imgfilenames = sorted([filename for _, _, filename in os.walk(trainpath)][0])
        self.targetfilenames = sorted([filename for _, _, filename in os.walk(targetpath)][0])
        self.joint_transform = transform_rule
    def __len__(self):
        return len(self.imgfilenames) 

    def __getitem__(self, key):

        _img = Image.open(self.trainpath + self.imgfilenames[key]).convert('RGB')
        _target = Image.open(self.targetpath + self.targetfilenames[key]).convert('RGB')

        if self.joint_transform is not None:
            _img, _target = self.joint_transform([_img, _target])
        #print(f"filenames called {_img.size()}:{_target.size()}")
        return _img, _target.permute(2,1,0)


number_of_classes = 2

trainset = LIPDataset(transform_rule=train_transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=20, shuffle=True, num_workers=4)

train_subset_sampler = torch.utils.data.sampler.SubsetRandomSampler(range(len(trainset)))
train_subset_loader = torch.utils.data.DataLoader(
    dataset=trainset, 
    batch_size=1,
    sampler=train_subset_sampler,
    num_workers=2
)

################################################################## Training script ################################################################################

fcn = resnet_dilated.Resnet9_8s(num_classes=number_of_classes)
fcn.cuda()
fcn.train()

criterion = nn.CrossEntropyLoss(size_average=False).cuda()
optimizer = optim.Adam(fcn.parameters(), lr=0.0001, weight_decay=0.0001)





for epoch in range(30):  # loop over the dataset multiple times

    running_loss = 0.0
    
    for i, data in enumerate(trainloader, 0):
        
        # get the inputs
        img, anno = data
        
        # We need to flatten annotations and logits to apply index of valid
        # annotations. All of this is because pytorch doesn't have tf.gather_nd()
        anno_flatten = flatten_annotations(anno)
        index = get_valid_annotations_index(anno_flatten, mask_out_value=255)
        anno_flatten_valid = torch.index_select(anno_flatten, 0, index)

        # wrap them in Variable
        # the index can be acquired on the gpu
        img, anno_flatten_valid, index = Variable(img.cuda()), Variable(anno_flatten_valid.cuda()), Variable(index.cuda())

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        logits = fcn(img)
        logits_flatten = flatten_logits(logits, number_of_classes=2)
        logits_flatten_valid = torch.index_select(logits_flatten, 0, index)
        
        loss = criterion(logits_flatten_valid, anno_flatten_valid)
        loss.backward()
        optimizer.step()

        """
        # print statistics
        running_loss += (loss.data[0] / logits_flatten_valid.size(0)) 
        if i % 2 == 1:
            
            
            loss_history.append(running_loss / 2)
            loss_iteration_number_history.append(loss_current_iteration)
            
            loss_current_iteration += 1
            
            loss_axis.lines[0].set_xdata(loss_iteration_number_history)
            loss_axis.lines[0].set_ydata(loss_history)

            loss_axis.relim()
            loss_axis.autoscale_view()
            loss_axis.figure.canvas.draw()
            
            running_loss = 0.0
        """