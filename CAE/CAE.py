import torch
import torchvision.models as models
import resnetDecoder
import clusteringLayer

class CAE(torch.nn.Module):

	def __init__(self, img_size):
		super(CAE, self).__init__()
		resnet = models.resnet18(pretrained=True);
		resnet.maxpool = torch.nn.Sequential()
		self.encoder = torch.nn.Sequential(*list(resnet.children())[:-2])
		self.decoder = resnetDecoder.resnetDecoder18(out_size=img_size)
		self.clustering = clusteringLayer.clusteringLayer(inplanes=512)

	def forward(self, x):
		self.code = self.encoder(x)
		#self.embedding = self.clustering(self.code) 
		self.embedding = self.code
		return self.decoder(self.code);
