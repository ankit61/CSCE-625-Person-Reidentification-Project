import torch
import resnet as models
import resnetDecoder

class CAE(torch.nn.Module):

	def __init__(self, latentSpaceDim = -1):
		super(CAE, self).__init__()
		resnet = models.resnet18(pretrained=True);
		resnet.maxpool = torch.nn.Sequential()
		self.encoder = torch.nn.Sequential(*list(resnet.children())[:-2])
		self.decoder = resnetDecoder.resnetDecoder18()

	def forward(self, x):
		self.code = self.encoder(x)
		return self.decoder(self.code);
