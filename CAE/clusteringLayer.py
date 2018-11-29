import torch
import torch.nn as nn
import torchvision.models as models

class clusteringLayer(torch.nn.Module):
	def __init__(self, inplanes):
		super(clusteringLayer, self).__init__()
		self.conv1 = nn.Conv2d(inplanes, 256, kernel_size=3, padding=1)
		self.bn1 = nn.BatchNorm2d(256)
		self.prelu1 = nn.PReLU()
		self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
		self.dropout = nn.Dropout() 
		self.bn2 = nn.BatchNorm2d(128)
		self.prelu2 = nn.PReLU()
		self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
		self.bn3 = nn.BatchNorm2d(64)
		self.prelu3 = nn.PReLU()
		self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
		self.prelu4 = nn.PReLU()

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def forward(self, x):
		x = self.conv1(x)
		x = self.prelu1(self.bn1(x))
		x = self.conv2(x)
		x = self.dropout(x)
		x = self.prelu2(self.bn2(x))
		x = self.conv3(x)
		x = self.prelu3(self.bn3(x))
		x = self.prelu4(self.conv4(x))
		
		return x
