import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from enum import Enum
from statistics import mean

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
				reconstructionLoss = 255 * F.l1_loss(pred, target)
				# (255*(pred - target)).pow(2).view(pred.size(0), pred.size(1), -1).max(2)[0].mean(0).sum()
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
							self.prevMean[curID] = mean
						
						self.ID2count[curID] += 1
						self.ID2sum[curID] += embedding[i]
					else:
						self.ID2count[curID] = 1
						self.prevMean[curID] = mean = self.ID2sum[curID] = embedding[i]

					intraclassDist = F.l1_loss(embedding[i], mean)
#					interclassDist = torch.zeros(1, requires_grad=True)
#					for key in self.prevMean.keys():
#						if key is not curID:
#							d = F.l1_loss(embedding[i], self.prevMean[key])
#							interclassDist = interclassDist + d / (len(self.prevMean) - 1) 

					clusteringLoss = clusteringLoss + intraclassDist #- interclassDist


			clusteringLoss = clusteringLoss / pred.size(0)
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
			self.prevMean = {}
		self.epoch += 1

