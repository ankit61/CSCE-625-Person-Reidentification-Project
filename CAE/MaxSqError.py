import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from enum import Enum

writer = SummaryWriter("/runs")

class LossType(Enum):
	RECONSTRUCTION = 1
	CLUSTERING = 2
	BOTH = 3

class MaxSqError:
	
	def __init__(self, regConst, lossType=LossType.CLUSTERING):
		self.ID2sum = {}
		self.ID2count = {}
		self.regConst = regConst
		self.batch = 0
		self.epoch = 0
		self.updateMeanInBatches = 130
		self.resetMeanInEpochs = 20
		self.prevMean= {}
		self.lossType = lossType
	
	def __call__(self, pred, code, embedding, target, ID):
		
		if(pred.dim() == 4 and (pred.size() == target.size() or self.lossType == LossType.CLUSTERING)):
			if(self.lossType == LossType.RECONSTRUCTION or self.lossType == LossType.BOTH):
				reconstructionLoss = (pred - target).pow(2).view(pred.size(0), pred.size(1), -1).max(2)[0].mean(0).sum()
			else:
				reconstructionLoss = torch.zeros(1)

			#regularizationLoss = self.regConst * torch.mean(torch.abs(code))#.mean(0).sum()
			
			#compute clustering loss
			clusteringLoss = torch.zeros(1, requires_grad=True)
			if(self.lossType == LossType.CLUSTERING or self.lossType == LossType.BOTH):
				for i in range(pred.size(0)):
					curID = int(ID[i])
					if(curID in self.ID2sum):
						
						if(self.batch % self.updateMeanInBatches == 0):
							for key in self.prevMean.keys():
								self.prevMean[key] = self.ID2sum[key] / self.ID2count[key]
						
						if(curID in self.prevMean):
							mean = self.prevMean[curID]
						else:
							mean = self.ID2sum[curID] / self.ID2count[curID]
						
						clusteringLoss = clusteringLoss + F.mse_loss(embedding[i], mean) 
						self.ID2count[curID] += 1
						self.ID2sum[curID] += embedding[i]
					else:
						self.ID2count[curID] = 1
						self.ID2sum[curID] = embedding[i]

			loss = clusteringLoss + reconstructionLoss #+ regularizationLoss
			
			if(self.lossType == LossType.BOTH or self.lossType == LossType.CLUSTERING):
				writer.add_scalar("lossCAE/clustering", clusteringLoss / loss, self.batch)
			if(self.lossType == LossType.BOTH or self.lossType == LossType.RECONSTRUCTION):
				writer.add_scalar("lossCAE/reconstruction", reconstructionLoss / loss, self.batch)
			#writer.add_scalar("/runs/cae/regularization", regularizationLoss / loss, self.batch)			
			self.batch += 1

			return loss

		else:
			raise Exception("pred and target must have 4D with same size.  size of pred: " + str(pred.size()) + " size of target: " + str(target.size()))

	def reset(self):
		if(self.epoch % self.resetMeanInEpochs == 0):
			self.ID2sum= {}
			self.ID2count = {}
		self.epoch += 1

