#!/usr/bin/env python3
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

g_writer = SummaryWriter("/runs/siamese")
MEAN		= [0.216, 0.2074816, 0.22934238]
STD			= [0.2333638, 0.22653223, 0.23671082]
INPUT_SIZE	= (208, 76)

g_parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
g_parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
					help='number of data loading workers (default: 4)')
g_parser.add_argument('--epochs', default=1000, type=int, metavar='N',
					help='number of total epochs to run')
g_parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
					help='manual epoch number (useful on restarts)')
g_parser.add_argument('-b', '--batch-size', default=64, type=int,
					metavar='N', help='mini-batch size (default: 64)')
g_parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
					metavar='LR', help='initial learning rate')
g_parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
					help='momentum')
g_parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
					metavar='W', help='weight decay (default: 5e-4)')
g_parser.add_argument('--print-freq', '-p', default=20, type=int,
					metavar='N', help='print frequency (default: 20)')
g_parser.add_argument('--resume', default='', type=str, metavar='PATH',
					help='path to latest checkpoint (default: none)')
g_parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
					help='evaluate model on validation set')
g_parser.add_argument('--save-dir', dest='save_dir',
					help='The directory used to save the trained models',
					default='trained_resnets', type=str)

def train(_train_loader, _model, _criterion, _optimizer, _epoch, _print_freq):
	_model.train()
	
	for i, (x, y, isSame) in enumerate(_train_loader):
		xVar = torch.autograd.Variable(x).cuda()
		yVar = torch.autograd.Variable(y).cuda()

		xNN, yNN = _model(xVar, yVar)
		loss = _criterion(xNN, yNN, isSame)

		_optimizer.zero_grad()
		loss.backward()
		_optimizer.step()

		if(i % _print_freq == 0):
			print('Epoch[' + str(_epoch) + '][' + str(i) + '/' + str(len(_train_loader)) + ']:\tLoss:', loss.item())

			g_writer.add_scalar('training_loss' , loss.item(), _epoch * len(_train_loader) + i)

def test(_test_loader, _model, _criterion, _epoch, _print_freq):
	"""
	_model.eval()

	with torch.no_grad():
		avgLoss = 0
		for i, (x, y, isSame) in enumerate(_test_loader):		
			xVar = torch.autograd.Variable(x).cuda()
			yVar = torch.autograd.Variable(y).cuda()

			xNN, yNN = _model(xVar, yVar)
			loss = _criterion(xNN, yNN, isSame)

			avgLoss += loss

			if(i % _print_freq == 0):
				print('Epoch[' + str(i) + '/' + str(len(_test_loader)) + ']:\tLoss: ' + str(loss))
				g_writer.add_scalar('val_loss' , loss.item(), _epoch * len(_test_loader) + i)

		avgLoss /= len(_test_loader)
	"""


	top1, top5, top10, ap = generateResults(
		"/datasets/TAMUvalSegmented/query/", 
		"/datasets/TAMUvalSegmented/gallery/", 
		_model
	)



	return avgLoss

def main():
	args		= g_parser.parse_args()
	model		= Siamese()
	criterion	= ContrastiveLoss()
	optimizer	= torch.optim.Adam(model.parameters(), args.lr)
	
	model.cuda()
	cudnn.benchmark = True
	best_loss = 1000

	normalize = transforms.Normalize(mean=MEAN, std=STD)

	train_dataset = SiameseDataset("/datasets/DukeSegmented/train/", transforms=transforms.Compose([
			transforms.RandomHorizontalFlip(),
			transforms.Resize(INPUT_SIZE),
			transforms.ToTensor(),
			normalize,
		]))

	train_loader = torch.utils.data.DataLoader(
		train_dataset,
		batch_size=args.batch_size,
		sampler = SiameseSampler(train_dataset),  
		num_workers=args.workers, pin_memory=True)
	
	val_dataset = SiameseDataset("/datasets/DukeSegmented/val/", transforms=transforms.Compose([
			transforms.Resize(INPUT_SIZE),
			transforms.ToTensor(),
			normalize,
		]))
	
	test_loader = torch.utils.data.DataLoader(
		train_dataset,
		batch_size=args.batch_size,
		sampler = SiameseSampler(val_dataset),
		num_workers=args.workers, pin_memory=True)
	
	if args.resume:
		if os.path.isfile(args.resume):
			print("=> loading checkpoint '{0}'".format(args.resume))
			checkpoint = torch.load(args.resume)
			#args.start_epoch = checkpoint['epoch']
			best_loss = checkpoint['loss']
			model.load(checkpoint['state_dict'])
			print("=> loaded checkpoint '{}' (epoch {})"
				  .format(args.evaluate, checkpoint['epoch']))
		else:
			print("=> no checkpoint found at '{}'".format(args.resume))

	for epoch in range(args.epochs):
		adjust_learning_rate(optimizer, epoch, args.lr)
		train(train_loader, model, criterion, optimizer, epoch, 20)

		loss = test(test_loader, model, criterion, epoch, 20)

		if(loss < best_loss or epoch % 20 == 0):
			best_loss = min(loss, best_loss)
			save_checkpoint({
				'epoch': epoch + 1,
				'state_dict': model.state_dict(),
				'loss': best_loss,
			}, True, filename=os.path.join(args.save_dir, 'checkpoint_{}.tar'.format(epoch)))

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
	torch.save(state, filename)

def adjust_learning_rate(optimizer, epoch, _lr):
	"""Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
	lr = _lr * (0.5 ** (epoch // 30))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

if __name__ == "__main__":
	main()
