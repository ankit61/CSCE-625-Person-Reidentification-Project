#!/usr/bin/env python3
import sys
import os
sys.path.append('../')
sys.path.append("/pytorch-segmentation/pytorch-segmentation-detection/")
sys.path.insert(0, '/pytorch-segmentation/pytorch-segmentation-detection/vision/')
from pytorch_segmentation_detection.transforms import RandomCropJoint
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix
import PIL
from PIL import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt
import random
import numbers
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torchvision
import torch
from pytorch_segmentation_detection.transforms import (
    ComposeJoint,
    RandomHorizontalFlipJoint,
    RandomScaleJoint,
    CropOrPad,
    ResizeAspectRatioPreserve
)
import pytorch_segmentation_detection.models.resnet_dilated as resnet_dilated
import pytorch_segmentation_detection.models.fcn as fcns
from pytorch_segmentation_detection.datasets.endovis_instrument_2017 import Endovis_Instrument_2017

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

#writer = SummaryWriter('/runs/images')
SIZE_H = 300
SIZE_W = 300

#defining global training and valdation 
resize_func = ScaleDownOrPad((SIZE_W, SIZE_H))

train_transform = ComposeJoint(
    [
        RandomHorizontalFlipJoint(),
        lambda inputs: [resize_func(inp) for inp in inputs],
        [transforms.ToTensor(), None],
        [transforms.Normalize((0.35070873, 0.3755584, 0.4201221),
                              (0.23140314, 0.23619365, 0.24928139)), None],
        [None, transforms.Lambda(
            lambda x: torch.from_numpy(np.asarray(x)).long())]
    ]
)

valid_mean = (0.35070873, 0.3755584, 0.4201221)
valid_stddev = (0.23140314, 0.23619365, 0.24928139)
valid_transform = ComposeJoint(
    [
        lambda inputs: [resize_func(inp) for inp in inputs],
        [transforms.ToTensor(), None],
        [transforms.Normalize(valid_mean, valid_stddev ), None],
        [None, transforms.Lambda(
            lambda x: torch.from_numpy(np.asarray(x)).long())]
    ]
)


def flatten_logits(logits, number_of_classes):
    """Flattens the logits batch except for the logits dimension"""

    logits_permuted = logits.permute(0, 2, 3, 1)
    logits_permuted_cont = logits_permuted.contiguous()
    logits_flatten = logits_permuted_cont.view(-1, number_of_classes)

    return logits_flatten


def flatten_annotations(annotations):

    return annotations.view(-1)


def get_valid_annotations_index(flatten_annotations, mask_out_value=255):

    return torch.squeeze(torch.nonzero((flatten_annotations != mask_out_value)), 1)

class LIPDataset(torch.utils.data.Dataset):
    def __init__(self, trainpath, targetpath, transform_rule=None):
        self.trainpath = trainpath
        self.targetpath = targetpath
        self.imgfilenames = sorted(
            [filename for _, _, filename in os.walk(trainpath)][0])
        self.targetfilenames = sorted(
            [filename for _, _, filename in os.walk(targetpath)][0])
        self.joint_transform = transform_rule

    def __len__(self):
        return len(self.imgfilenames)

    def __getitem__(self, key):

        _img = Image.open(
            self.trainpath + self.imgfilenames[key]).convert('RGB')
        _target = Image.open(
            self.targetpath + self.targetfilenames[key]).convert('RGB')

        if self.joint_transform is not None:
            _img, _target = self.joint_transform([_img, _target])

        # print(_img.size())

        _target = _target.permute(2, 1, 0)[0:1, :, :]

        #print(f"filenames called {_img.size()}:{_target.size()}")
        return _img, _target

class TestDataset(torch.utils.data.Dataset):
    def __init__( self, inputpath, output_path):
        self.inputpath = inputpath
        self.output_path = output_path
        self.imgfilenames = sorted(
            [filename for _, _, filename in os.walk(inputpath)][0])

    def __len__(self):
        return len(self.imgfilenames)

    def __getitem__(self, key):
        
        
        _img = Image.open(
            self.inputpath + self.imgfilenames[key])
        
        _img, _ = valid_transform([_img, _img])

        # convert to tensor to be used by pytorch
        
        #_img = _img.permute(2, 1, 0)
        #print(_img.shape)
        return _img, self.imgfilenames[key]

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
                oh = int(self.greater_side_size * h / w)
                input = input.resize((ow, oh), self.interpolation)

            else:
                oh = self.greater_side_size
                ow = int(self.greater_side_size * w / h)
                input = input.resize((ow, oh), self.interpolation)

        input_position = (np.asarray(self.output_size) // 2) - \
            (np.asarray(input.size) // 2)

        output = Image.new(mode=input.mode,
                           size=self.output_size,
                           color=self.fill)

        output.paste(input, box=tuple(input_position))

        return output

class SegmentaionNetwork():
    def __init__(self, weights=None, number_of_classes=2):
        self.fcn = resnet_dilated.Resnet18_8s(num_classes=number_of_classes)
        if weights:
            fcn.load_state_dict(torch.load(weights))
       
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        return
    
    def eval(self):
        self.fcn.eval()
    
    def train(self):
        self.fcn.train()

# Uncomment to preserve BN statistics
# fcn.eval()
# for m in fcn.modules():

#     if isinstance(m, nn.BatchNorm2d):
#         m.weight.requires_grad = False
#         m.bias.requires_grad = False

# Define the loss and load it to gpu
#optimizer = optim.Adam(filter(lambda p: p.requires_grad, fcn.parameters()), lr=0.00001, weight_decay=0.0005)

    
def process_images(dataset_dir, processed_dir):

    dataset = TestDataset(inputpath=dataset_dir, output_path=processed_dir)
    images_to_process = torch.utils.data.DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=1
    )

    fcn.eval()
    num_of_images = len(images_to_process)
    count = 0
    for image, name in images_to_process:
        gt_image = np.array(image.cpu()[0].type(torch.FloatTensor).numpy())
        image = Variable(image.cuda())

        #image = Variable(image)
        logits = fcn(image)

        # First we do argmax on gpu and then transfer it to cpu
        logits = logits.data
        _, prediction = logits.max(1)
        prediction = prediction.squeeze(1)
        
        prediction_for_output = transforms.ToPILImage()(prediction.cpu().permute(0, 2, 1).type(torch.FloatTensor)).convert('L')
        #prediction_for_output = prediction_for_output[:, :, 0:1].copy()
        gt_image[0, :, :] = gt_image[0, :, :] * valid_stddev[0] + valid_mean[0]
        gt_image[1, :, :] = gt_image[1, :, :] * valid_stddev[1] + valid_mean[1]
        gt_image[2, :, :] = gt_image[2, :, :] * valid_stddev[2] + valid_mean[2]

        f = np.vectorize(lambda x: np.uint8(x*255))
        gt_image = f(gt_image).copy()
        gt_image = np.transpose(gt_image, (1,2,0))
        gt_image = transforms.ToPILImage()(gt_image).convert('RGB')

        #gt_image = gt_image[:, :, :].copy()
        #prediction_for_output= np.transpose(prediction_for_output, (2, 0, 1) )

        #print(gt_image.size)
        #print(prediction_for_output.size)

        #masked = cv2.bitwise_and(prediction_for_output, gt_image, mask=gt_image)
        
        comp = Image.new('RGB', (SIZE_W, SIZE_H))
        comp.paste(gt_image, mask=prediction_for_output)
        

        #cropping image to save on batch size in CAE
        # help from https://stackoverflow.com/questions/14211340/automatically-cropping-an-image-with-python-pil/42123638#42123638
        comp_array = np.asarray(comp)
        comp_bw = comp_array.max(axis=2)
        non_empty_columns = np.where(comp_bw.max(axis=0)>0)[0]
        non_empty_rows = np.where(comp_bw.max(axis=1)>0)[0]
        cropBox = (min(non_empty_rows), max(non_empty_rows), min(non_empty_columns), max(non_empty_columns))

        comp_reduced = comp_array[cropBox[0]:cropBox[1]+1, cropBox[2]:cropBox[3]+1 , :]

        new_comp = Image.fromarray(comp_reduced)


        new_comp.save(processed_dir + name[0])

        #writer.add_image('DUKESegmented/Image' + str(count) + '/Predicted', prediction_for_output, count)
        #writer.add_image('DUKESegmented/Image' + str(count) + '/Original', gt_image, count)
        #cv2.imwrite(processed_dir + name[0], np.transpose(masked, (1,2,0)))

        count += 1

# Define the validation function to track MIoU during the training


def validate(fcn, imagepath, targetpath, number_of_classes=2):
    valset = LIPDataset(
        transform_rule=valid_transform,
        trainpath=imagepath,
        targetpath=targetpath
    )
    val_subset_sampler = torch.utils.data.sampler.SubsetRandomSampler(range(1000))
    valset_loader = torch.utils.data.DataLoader(
        valset, 
        batch_size=1, 
        sampler=val_subset_sampler,
        shuffle=False, 
        num_workers=2
    )
    
    labels = range(number_of_classes)
    fcn.eval()
    overall_confusion_matrix = None

    l_valset = len(valset_loader)

    print(l_valset)

    count = 0

    for image, annotation in valset_loader:
        # Store raw image
        gt_image = np.array(image.cpu()[0].type(torch.FloatTensor).numpy())
        # Reshape the annotation image into the orignial size
        annotation_image = np.array(annotation.cpu()[0].permute(0, 2, 1).type(torch.FloatTensor).numpy())
        image = Variable(image.cuda())

        #image = Variable(image)
        logits = fcn(image)

        # First we do argmax on gpu and then transfer it to cpu
        logits = logits.data
        _, prediction = logits.max(1)
        prediction = prediction.squeeze(1)


        # ----------------- Tensor Board Image Output --------------------------
        # Move the prediction to the cpu, convert it to a tensor, and then to a PIL image
        prediction_for_output = np.array(prediction.cpu().permute(0, 2, 1).type(torch.FloatTensor).numpy())
        
        # 
        
        # Denormalize the original image (channelwise * std + mean )
        gt_image[ 0, :, :] = gt_image[0, :, :] * valid_stddev[0] + valid_mean[0]
        gt_image[1, :, :] = gt_image[1, :, :] * valid_stddev[1] + valid_mean[1]
        gt_image[2, :, :] = gt_image[2, :, :] * valid_stddev[2] + valid_mean[2]

        # Add images to tensorboard
        writer.add_image('SegmentationResults/Image' + str(count) + '/Predicted', prediction_for_output, count)
        writer.add_image('SegmentationResults/Image' + str(count) + '/Ground Truth', annotation_image, count)
        writer.add_image('SegmentationResults/Image' + str(count) + '/Original', gt_image, count)
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
    union = ground_truth_set + predicted_set - intersection

    intersection_over_union = intersection / union.astype(np.float32)
    mean_intersection_over_union = np.mean(intersection_over_union)

    fcn.train()

    return mean_intersection_over_union

def start_training(
    train_imgpath,
    train_targetpath,
    validate_imgpath,
    validate_targetpath,
    epochs=200
):
    best_validation_score = 0

    trainset = LIPDataset(train_imgpath, train_targetpath, transform_rule=train_transform)

    trainloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=70,
        shuffle=True, 
        num_workers=4
    )

    #First arument defines the weighted loss we are using for the function
    criterion = nn.CrossEntropyLoss(torch.Tensor([1.0, 3.0]), size_average=False).cuda()
    optimizer = optim.Adam(fcn.parameters(), lr=0.0001, weight_decay=0.0001)

    iter_size = 20
    with open("logfile10.txt", "a+") as file:
        with SegmentaionNetwork() as fcn:
            for epoch in range(0, epochs):  # loop over the dataset multiple times

                print(f"Epoch {epoch}")
                l_epoch = len(trainloader) - 1
                running_loss = 0.0
                for i, data in enumerate(trainloader, 0):
                    
                    # get the inputs
                    img, anno = data

                    # We need to flatten annotations and logits to apply index of valid
                    # annotations. All of this is because pytorch doesn't have tf.gather_nd()
                    anno_flatten = flatten_annotations(anno)
                    index = get_valid_annotations_index(
                        anno_flatten, mask_out_value=255)
                    anno_flatten_valid = torch.index_select(anno_flatten, 0, index)

                    # wrap them in Variable
                    # the index can be acquired on the gpu
                    img, anno_flatten_valid, index = Variable(img.cuda()), Variable(
                        anno_flatten_valid.cuda()), Variable(index.cuda())
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
                        file.write(
                            f"epoch {epoch} : {i}/{l_epoch} -> {running_loss / 5}")

                        avg_loss = running_loss / 5
                        writer.add_scalar('segmentation/total_loss' +
                                        str(epoch), avg_loss, i)
                        running_loss = 0.0

                current_validation_score = validate(fcn, validate_imgpath, validate_targetpath)
                print(f"TOTAL MIoU {current_validation_score}")
                file.write(f"TOTAL MIoU {current_validation_score}")

                # Save the model if it has a better MIoU score.
                if current_validation_score > best_validation_score:

                    torch.save(fcn.state_dict(),
                            f'resnet_18_8s_best_hsv_IMAGES{epoch}.pth')
                    best_validation_score = current_validation_score

    print('Finished Training')


# Process x Dataset



# Train on LIP Dataset
# start_training()
# process_images("/datasets/DukeMTMC-reID/bounding_box_test/", "/datasets/DukeSegmented/test/")
