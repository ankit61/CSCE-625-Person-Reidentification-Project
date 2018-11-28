#!/usr/bin/env python3
from CAE import CAE
import sys
import matplotlib.pyplot as plt
import torch
from tensorboardX import SummaryWriter
from PReIDDataset import PReIDDataset
from PReIDDataset import DatasetType
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torchvision.utils as utils
import torch.utils.data
import numpy as np

writer = SummaryWriter("/runs/")

img_size = (208, 76)
mean = [0.216, 0.2074816, 0.22934238]
std = [0.2333638, 0.22653223, 0.23671082]

def test(val_loader, model):
	"""
	Run evaluation
	"""
	# switch to evaluate mode
	model.eval()
	with torch.no_grad():
		toPIL = transforms.ToPILImage()
		for i, (input_img, target, ID) in enumerate(val_loader):
			target = target.cuda(async=True)
			input_var = torch.autograd.Variable(input_img, volatile=True).cuda()
			target_var = torch.autograd.Variable(target, volatile=True)

			# compute output
			output = model(input_var)
			#output = output.cpu().type(torch.FloatTensor).numpy()
			output[:, 0, :, :] = output[:, 0, :, :] * std[0] + mean[0]
			output[:, 1, :, :] = output[:, 1, :, :] * std[1] + mean[1]
			output[:, 2, :, :] = output[:, 2, :, :] * std[2] + mean[2]
			output = output.cpu()
						
			input_img[:, 0, :, :] = input_img[:, 0, :, :] * std[0] + mean[0]
			input_img[:, 1, :, :] = input_img[:, 1, :, :] * std[1] + mean[1]
			input_img[:, 2, :, :] = input_img[:, 2, :, :] * std[2] + mean[2]
			input_img = input_img.cpu()
			
			pilImg = F.to_pil_image(output)
			pilImg.save('segmented','.png')
			break;
			
			img = utils.make_grid(torch.cat((input_img, output), 0), nrow=2)
			writer.add_image("cae/im_" + str(i), img, i)

def main():
	model_path = sys.argv[1]
	checkpoint = torch.load(model_path)
	model = CAE(img_size, should_decode=True)
	model.cuda()
	model.load_state_dict(checkpoint['state_dict'])
	
	val_loader = torch.utils.data.DataLoader(
			PReIDDataset(DatasetType.VAL, transform=transforms.Compose([
				transforms.Resize(img_size),
				transforms.ToTensor(),
				transforms.Normalize(mean=mean, std=std),
			])),
			batch_size=1, shuffle=False,
			num_workers=4, pin_memory=True)

	test(val_loader, model)