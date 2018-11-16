import torch
import torch.nn as nn

class BasicDecoderBlock(nn.Module):
	expansion = 1

	def __init__(self, inplanes, planes, stride=1, upsample=None):
		super(BasicDecoderBlock, self).__init__()
		self.conv1 = nn.ConvTranspose2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)	  
		self.bn1 = nn.BatchNorm2d(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = nn.ConvTranspose2d(planes, planes, kernel_size=3, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.upsample = upsample
		self.stride = stride

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.upsample is not None:
			residual = self.upsample(x)

		out += residual
		out = self.relu(out)

		return out

class ResNetDecoder(nn.Module):

	def __init__(self, block, layers, out_size = (182, 74)):
		super(ResNetDecoder, self).__init__()
		self.inplanes = 512
		self.out_size = out_size
		self.conv1 = nn.ConvTranspose2d(64, 3, kernel_size=7, stride=2, padding=3, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.layer1 = self._make_layer(block, 512, layers[0])
		self.layer2 = self._make_layer(block, 256, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 64, layers[3], stride=2)

		for m in self.modules():
			if isinstance(m, nn.ConvTranspose2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			if isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def _make_layer(self, block, planes, blocks, stride=1):
		upsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			upsample = nn.Sequential(
				nn.ConvTranspose2d(self.inplanes, planes * block.expansion,
						  kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, upsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		
		x = self.bn1(x)
		x = self.conv1(x)

		return torch.nn.functional.upsample_bilinear(x, self.out_size)

def resnetDecoder9(**kwargs):
	return ResNetDecoder(BasicDecoderBlock, [1, 1, 1, 1], **kwargs)

def resnetDecoder18(**kwargs):
	return ResNetDecoder(BasicDecoderBlock, [2, 2, 2, 2], **kwargs)
