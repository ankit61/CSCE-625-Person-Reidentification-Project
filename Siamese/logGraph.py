from network import Siamese
import torchvision
import torch
import argparse
import os
import torchvision.transforms as transforms
from dataset import SiameseDataset
from dataset import SiameseSampler
import torch.backends.cudnn as cudnn
from network import Siamese
from contrastiveLoss import ContrastiveLoss
from tensorboardX import SummaryWriter
from mAP import generateResults

from tensorboardX import SummaryWriter


g_writer = SummaryWriter("/runs")

model = Siamese()
dummy_input = torch.autograd.Variable(torch.rand(1, 3, 128, 64))
dummy_input2 = torch.autograd.Variable(torch.rand(1, 3, 128, 64))


g_writer.add_graph(model, (dummy_input, dummy_input2))
g_writer.add_scalar("TestScalars/points", 5, 5)
print('Log Completed')
