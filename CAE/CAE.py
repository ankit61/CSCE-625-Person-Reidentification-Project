import torch
import torchvision.models as models
import resnetDecoder
import clusteringLayer

class CAE(torch.nn.Module):

	def __init__(self, img_size, should_decode=True, freeze_encoder=True):
		super(CAE, self).__init__()
		
		resnet = models.resnet18(pretrained=True);
		resnet.maxpool = torch.nn.Sequential()
		
		self.encoder = torch.nn.Sequential(*list(resnet.children())[:-2])
		if(freeze_encoder):
			for param in self.encoder.parameters():
				param.requires_grad = False
		
		if(should_decode):
			self.decoder = resnetDecoder.resnetDecoder18(out_size=img_size)
			self.clustering = torch.nn.Sequential()
		else:
			self.decoder = torch.nn.Sequential()
			self.clustering = clusteringLayer.clusteringLayer(inplanes=512)

	def forward(self, x):
		self.code = self.encoder(x)
		self.embedding = self.clustering(self.code) 
		return self.decoder(self.code);

	def load(self, state_dict):
		model_state_dict = self.state_dict()
		sub_dict = {k : v for k, v in state_dict.items() if k in model_state_dict and v.size() == model_state_dict[k].size() }
		model_state_dict.update(sub_dict)
		self.load_state_dict(model_state_dict)
