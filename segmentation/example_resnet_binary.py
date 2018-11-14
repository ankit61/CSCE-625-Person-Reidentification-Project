#!/usr/bin/env python3
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('matplotlib', 'notebook')

import sys, os
sys.path.append("/pytorch-segmentation/pytorch-segmentation-detection/")
sys.path.insert(0, '/pytorch-segmentation/pytorch-segmentation-detection/vision/')

# Use second GPU -pytorch-segmentation-detection- change if you want to use a first one
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from pytorch_segmentation_detection.datasets.endovis_instrument_2017 import Endovis_Instrument_2017
import pytorch_segmentation_detection.models.fcn as fcns
import pytorch_segmentation_detection.models.resnet_dilated as resnet_dilated
from pytorch_segmentation_detection.transforms import (ComposeJoint,
                                                       RandomHorizontalFlipJoint,
                                                       RandomScaleJoint,
                                                       CropOrPad,
                                                       ResizeAspectRatioPreserve)


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

# Tensorboard 
from tensorboardX import SummaryWriter

writer = SummaryWriter('/runs/images')

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


from pytorch_segmentation_detection.transforms import RandomCropJoint

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
        
        #print(_img.size())

        _target = _target.permute(2,1,0)[0:1,:,:]

        #print(f"filenames called {_img.size()}:{_target.size()}")
        return _img, _target

class ScaleDownOrPad(object):
    
    
    def __init__(self, output_size, interpolation=Image.BILINEAR, fill=0):
        self.interpolation = interpolation
        self.fill = fill
        self.output_size = output_size
        if output_size[0] > output_size[1]:
            self.greater_side_size = output_size[0]
        else:
            self.greater_side_size = output_size[1]
        
    def __call__(self, input):
        w, h = input.size
        if w > self.output_size[0] or h > self.output_size[1]:
            if w > h:
                ow = self.greater_side_size
                oh = int( self.greater_side_size * h / w )
                input = input.resize((ow, oh), self.interpolation)

            else:
                oh = self.greater_side_size
                ow = int(self.greater_side_size * w / h)
                input =  input.resize((ow, oh), self.interpolation)
        
        input_position = (np.asarray(self.output_size) // 2) - (np.asarray(input.size) // 2)

        output = Image.new(mode=input.mode,
                           size=self.output_size,
                           color=self.fill)
        
        output.paste(input, box=tuple(input_position))
        
        return output


number_of_classes = 2

labels = range(number_of_classes)

resize_func = ScaleDownOrPad((244,244))

train_transform = ComposeJoint(
                [
                    RandomHorizontalFlipJoint(),
                    lambda inputs: [resize_func(inp) for inp in inputs],
                    [transforms.ToTensor(), None],
                    [transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), None],
                    [None, transforms.Lambda(lambda x: torch.from_numpy(np.asarray(x)).long()) ]
                ])

trainset = LIPDataset(transform_rule=train_transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                          shuffle=True, num_workers=4)



valid_transform = ComposeJoint(
                [
                     lambda inputs: [resize_func(inp) for inp in inputs],
                     [transforms.ToTensor(), None],
                     [transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), None],
                     [None, transforms.Lambda(lambda x: torch.from_numpy(np.asarray(x)).long()) ]
                ])


valset = LIPDataset(
    transform_rule=valid_transform, 
    trainpath = "/datasets/LIP/TrainVal_images/val_images/", 
    targetpath = "/datasets/LIP/EDITED_TrainVal_parsing_annotations/val_segmentations/"
)


val_subset_sampler = torch.utils.data.sampler.SubsetRandomSampler(range(1000))

valset_loader = torch.utils.data.DataLoader(valset, batch_size=1, sampler=val_subset_sampler,
                                            shuffle=False, num_workers=2)

# Define the validation function to track MIoU during the training
def validate():
    
    fcn.eval()
    
    overall_confusion_matrix = None

    l_valset = len(valset_loader)

    print(l_valset)

    count = 0

    for image, annotation in valset_loader:
        
        gt_image = image
        
        image = Variable(image.cuda())
        
        #image = Variable(image)
        logits = fcn(image)

        # First we do argmax on gpu and then transfer it to cpu
        logits = logits.data
        _, prediction = logits.max(1)
        prediction = prediction.squeeze(1)
        


        
        # ----------------- Tensor Board Image Output --------------------------
        # Move the prediction to the cpu
        prediction_for_output = prediction.cpu()
        # Convert to a numpy array from a torch Float Tensor
        prediction_for_output = prediction_for_output.type(torch.FloatTensor).numpy()

        # Add images to tensorboard
        writer.add_image('SegmentationResults/Image' + str(count) + '/Predicted', prediction_for_output, count)
        writer.add_image('SegmentationResults/Image' + str(count) + '/Ground Truth Labeled', annotation, count)
        writer.add_image('SegmentationResults/Image' + str(count) + '/Ground Truth', gt_image, count)
        print('Logged Image #' + str(count))
        # ----------------------------------------------------------------------
        prediction_np = prediction.cpu().numpy().flatten()
        annotation_np = annotation.numpy().flatten()

        # Mask-out value is ignored by default in the sklearn
        # read sources to see how that was handled

        current_confusion_matrix = confusion_matrix(y_true=annotation_np,
                                                    y_pred=prediction_np,
                                                    labels=labels)

        if overall_confusion_matrix is None:


            overall_confusion_matrix = current_confusion_matrix
        else:

            overall_confusion_matrix += current_confusion_matrix

        count += 1
        if count % 100 == 0:
            print(f"validating {count}/1000")
        
    
    
     

    intersection = np.diag(overall_confusion_matrix)
    ground_truth_set = overall_confusion_matrix.sum(axis=1)
    predicted_set = overall_confusion_matrix.sum(axis=0)
    union =  ground_truth_set + predicted_set - intersection

    intersection_over_union = intersection / union.astype(np.float32)
    mean_intersection_over_union = np.mean(intersection_over_union)
    
    fcn.train()

    return mean_intersection_over_union

"""
def validate_train():
    
    fcn.eval()
    
    overall_confusion_matrix = None

    for image, annotation in train_subset_loader:

        image = Variable(image)
        image = Variable(image.cuda())
        logits = fcn(image)

        # First we do argmax on gpu and then transfer it to cpu
        logits = logits.data
        _, prediction = logits.max(1)
        prediction = prediction.squeeze(1)

        prediction_np = prediction.cpu().numpy().flatten()
        annotation_np = annotation.numpy().flatten()

        # Mask-out value is ignored by default in the sklearn
        # read sources to see how that was handled

        current_confusion_matrix = confusion_matrix(y_true=annotation_np,
                                                    y_pred=prediction_np,
                                                    labels=labels)

        if overall_confusion_matrix is None:


            overall_confusion_matrix = current_confusion_matrix
        else:

            overall_confusion_matrix += current_confusion_matrix
    
    
    intersection = np.diag(overall_confusion_matrix)
    ground_truth_set = overall_confusion_matrix.sum(axis=1)
    predicted_set = overall_confusion_matrix.sum(axis=0)
    union =  ground_truth_set + predicted_set - intersection

    intersection_over_union = intersection / union.astype(np.float32)
    mean_intersection_over_union = np.mean(intersection_over_union)
    
    fcn.train()

    return mean_intersection_over_union
"""


## Define the model and load it to the gpu
fcn = resnet_dilated.Resnet18_8s(num_classes=2)
fcn.load_state_dict(torch.load("./resnet_18_8s_best_hsv29.pth"))
fcn.cuda()
fcn.train()

# Uncomment to preserve BN statistics
#fcn.eval()
# for m in fcn.modules():

#     if isinstance(m, nn.BatchNorm2d):
#         m.weight.requires_grad = False
#         m.bias.requires_grad = False

## Define the loss and load it to gpu
#optimizer = optim.Adam(filter(lambda p: p.requires_grad, fcn.parameters()), lr=0.00001, weight_decay=0.0005)

criterion = nn.CrossEntropyLoss(torch.Tensor([1.0,3.0]), size_average=False).cuda()


optimizer = optim.Adam(fcn.parameters(), lr=0.0001, weight_decay=0.0001)


best_validation_score = 0

iter_size = 20
with open("logfile10.txt", "a+") as file:
    for epoch in range(2,200):  # loop over the dataset multiple times
        
        print(f"Epoch {epoch}")
        l_epoch = len(trainloader)
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            current_validation_score = validate()
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
            #img, anno_flatten_valid, index = Variable(img, Variable(anno_flatten_valid), Variable(index))

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            logits = fcn(img)
            logits_flatten = flatten_logits(logits, number_of_classes=2)
            logits_flatten_valid = torch.index_select(logits_flatten, 0, index)
            
            loss = criterion(logits_flatten_valid, anno_flatten_valid)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += (loss.data[0] / logits_flatten_valid.size(0)) 
            if (i + 1) % 5 == 0:
                
                print(f"epoch {epoch} : {i}/{l_epoch} -> {running_loss / 5}")
                file.write(f"epoch {epoch} : {i}/{l_epoch} -> {running_loss / 5}")
                
                avg_loss = running_loss / 5
                writer.add_scalar('segmentation/total_loss' + str(epoch), avg_loss, i)
                running_loss = 0.0
                
            
                
        current_validation_score = validate()

        print(f"TOTAL MIoU {current_validation_score}")
        file.write(f"TOTAL MIoU {current_validation_score}")
                


        # Save the model if it has a better MIoU score.
        if current_validation_score > best_validation_score:

            torch.save(fcn.state_dict(), f'resnet_18_8s_best_hsv_IMAGES{epoch}.pth')
            best_validation_score = current_validation_score
                

print('Finished Training')

