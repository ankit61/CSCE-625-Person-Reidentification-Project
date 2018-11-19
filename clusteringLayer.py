import torch
import torch.nn as nn

class clusteringLayer(torch.nn.Module):
	def __init__(self, inplanes):
		super(clusteringLayer, self).__init__()
		self.conv1 = nn.Conv2d(inplanes, 128, kernel_size=3, padding=1)
		self.bn1 = nn.BatchNorm2d(128)
		self.prelu1 = nn.PReLU()
		self.conv2 = nn.Conv2d(128, 32, kernel_size=3, padding=1);
		self.bn2 = nn.BatchNorm2d(32)
		self.prelu2 = nn.PReLU()
		self.conv3 = nn.Conv2d(32, 3, kernel_size=3, padding=1)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)


	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.prelu1(x)
		x = self.conv2(x)
		x = self.bn2(x)
		x = self.prelu2(x)
		x = self.conv3(x)
		return x;
