import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class Siamese(torch.nn.Module):
	def __init__(self):
		super(Siamese, self).__init__()
		self.resnet = models.resnet101(pretrained=True)
		self.resnet.fc = nn.Sequential()
		self.resnet.avgpool = nn.Sequential()

	def forward(self, x, y):
		o1 = self.resnet(x)
		o2 = self.resnet(y)
		return o1, o2

	def forwardForOne(self, x):
		return self.resnet(x)

	def load(self, state_dict):
		model_state_dict = self.state_dict()
		sub_dict = {k : v for k, v in state_dict.items() if k in model_state_dict and v.size() == model_state_dict[k].size() }
		model_state_dict.update(sub_dict)
		self.load_state_dict(model_state_dict)
