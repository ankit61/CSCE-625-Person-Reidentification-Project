#!/usr/bin/env python3
import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from PReIDDataset import PReIDDataset
from PReIDDataset import DatasetType
from CAE import CAE
from tensorboardX import SummaryWriter

writer = SummaryWriter("/runs/")

torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)

input_size = (218, 85)
mean = [0.43993556, 0.43089762, 0.44579148]
std = [0.19977007, 0.20279744, 0.19051233]

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg19_bn')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
					help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
					help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
					help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
					metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
					metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
					help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
					metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=20, type=int,
					metavar='N', help='print frequency (default: 20)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
					help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
					help='evaluate model on validation set')
parser.add_argument('--save-dir', dest='save_dir',
					help='The directory used to save the trained models',
					default='trained_resnets', type=str)
parser.add_argument('--test-freq', dest='test_freq',
					help='number of epochs to test on entire testing dataset',
					type=int, default=20000)
parser.add_argument('--reg-const', dest='reg_const',
					help='regularization const encouraging sparsity',
					type=int, default=1)

best_loss = 200

def MaxSqError(pred, model, target):
	if(pred.size() == target.size() and pred.dim() == 4):
		diff = (pred - target).pow(2).view(pred.size(0), pred.size(1), -1).max(2)[0]
		return diff.sum() + args.reg_const *  torch.mean(torch.abs(model.code))#.mean(0).sum()
	else:
		raise Exception("pred and target must have 4D with same size")


def main():
	global args, best_loss
	args = parser.parse_args()

	# Check the save_dir exists or not
	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)

	model = CAE(input_size)

	# model.features = torch.nn.DataParallel(model.features)
	model.cuda()

	# optionally resume from a checkpoint
	if args.resume:
		if os.path.isfile(args.resume):
			print("=> loading checkpoint '{0}'".format(args.resume))
			checkpoint = torch.load(args.resume)
			args.start_epoch = checkpoint['epoch']
			best_loss = checkpoint['loss']
			model.load_state_dict(checkpoint['state_dict'])
			print("=> loaded checkpoint '{}' (epoch {})"
				  .format(args.evaluate, checkpoint['epoch']))
		else:
			print("=> no checkpoint found at '{}'".format(args.resume))

	cudnn.benchmark = True

	normalize = transforms.Normalize(mean=mean,
									 std=std)

	train_loader = torch.utils.data.DataLoader(
		PReIDDataset(DatasetType.TRAIN, transform=transforms.Compose([
			transforms.RandomHorizontalFlip(),
			 transforms.Resize(input_size),
			transforms.ToTensor(),
			normalize,
		])),
		batch_size=args.batch_size, shuffle=True,
		num_workers=args.workers, pin_memory=True)

	val_loader = torch.utils.data.DataLoader(
		PReIDDataset(DatasetType.VAL, transform=transforms.Compose([
			 transforms.Resize(input_size),
			transforms.ToTensor(),
			normalize,
		])),
		batch_size=args.batch_size, shuffle=False,
		num_workers=args.workers, pin_memory=True)

	test_loader = torch.utils.data.DataLoader(
		PReIDDataset(DatasetType.TEST, transform=transforms.Compose([
			 transforms.Resize(input_size),
			transforms.ToTensor(),
			normalize,
		])),
		batch_size=args.batch_size, shuffle=False,
		num_workers=args.workers, pin_memory=True)

	# define loss function (criterion) and pptimizer
	criterion = MaxSqError#nn.MSELoss().cuda()

	optimizer = torch.optim.Adam(model.parameters(), args.lr)

	if args.evaluate:
		validate(val_loader, model, criterion)
		return

	for epoch in range(args.start_epoch, args.epochs):
		adjust_learning_rate(optimizer, epoch)

		# train for one epoch
		train(train_loader, model, criterion, optimizer, epoch)

		# evaluate on validation set
		loss = validate(val_loader, model, criterion)
		
		writer.add_scalar('/runs/cae/validationLoss' , loss.item(), epoch)

		# remember best prec@1 and save checkpoint
		if(loss < best_loss):
			best_loss = min(loss, best_loss)
			save_checkpoint({
				'epoch': epoch + 1,
				'state_dict': model.state_dict(),
				'loss': best_loss,
			}, True, filename=os.path.join(args.save_dir, 'checkpoint_{}.tar'.format(epoch)))

		if(epoch % args.test_freq == 0 and epoch > args.start_epoch):
			validate(test_loader, model, criterion) 

def train(train_loader, model, criterion, optimizer, epoch):
	"""
		Run one train epoch
	"""
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()

	# switch to train mode
	model.train()

	end = time.time()
	for i, (input, target) in enumerate(train_loader):

		# measure data loading time
		data_time.update(time.time() - end)

		target = target.cuda(async=True)
		input_var = torch.autograd.Variable(input).cuda()
		target_var = torch.autograd.Variable(target)

		# compute output
		output = model(input_var)
		loss = criterion(output, model, target_var)

		# compute gradient and do SGD step
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		output = output.float()
		loss = loss.float()
		# measure accuracy and record loss
		losses.update(loss.data[0], input.size(0))

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if i % args.print_freq == 0:
			print('Epoch: [{0}][{1}/{2}]\t'
				  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
				  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
					  epoch, i, len(train_loader), batch_time=batch_time,
					  data_time=data_time, loss=losses))
			writer.add_scalar('/runs/cae/all' , loss.item(), epoch * len(train_loader) + i)

def validate(val_loader, model, criterion):
	"""
	Run evaluation
	"""
	batch_time = AverageMeter()
	losses = AverageMeter()

	# switch to evaluate mode
	model.eval()

	end = time.time()
	for i, (input, target) in enumerate(val_loader):
		target = target.cuda(async=True)
		input_var = torch.autograd.Variable(input, volatile=True).cuda()
		target_var = torch.autograd.Variable(target, volatile=True)

		# compute output
		output = model(input_var)
		loss = criterion(output, model, target_var)

		output = output.float()
		loss = loss.float()

		# measure accuracy and record loss
		losses.update(loss.data[0], input.size(0))

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if i % args.print_freq == 0:
			print('Test: [{0}/{1}]\t'
				  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				  .format(i, len(val_loader), batch_time=batch_time, loss=losses))

	return losses.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
	"""
	Save the training model
	"""
	torch.save(state, filename)

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
	"""Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
	#if(epoch > 10):
	#	lr = args.lr * 1e-8 * (0.5 ** (epoch // 30))
	#else:
	lr = args.lr * (0.5 ** (epoch // 30))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

if __name__ == '__main__':
	main()
