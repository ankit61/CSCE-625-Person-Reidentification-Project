import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter

writer = SummaryWriter("/runs")

class MaxSqError:
	
	def __init__(self, regConst):
		self.ID2sum = {}
		self.ID2count = {}
		self.regConst = regConst
		self.batch = 0
		self.epoch = 0
		self.prevMean= {}
	
	def __call__(self, pred, code, embedding, target, ID):
		if(pred.size() == target.size() and pred.dim() == 4):
			reconstructionLoss = F.l1_loss(pred, target)#, reduction='sum')
			#regularizationLoss = self.regConst * torch.mean(torch.abs(code))#.mean(0).sum()
			
			#compute clustering loss
			clusteringLoss = torch.zeros(1, requires_grad=True)
			for i in range(pred.size(0)):
				curID = int(ID[i])
				if(curID in self.ID2sum):
					if(self.batch % 130 == 0):
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
			
			writer.add_scalar("lossCAE/clustering", clusteringLoss / loss, self.batch)
			writer.add_scalar("lossCAE/reconstruction", reconstructionLoss / loss, self.batch)
			#writer.add_scalar("/runs/cae/regularization", regularizationLoss / loss, self.batch)			
			self.batch += 1

			return loss

		else:
			raise Exception("pred and target must have 4D with same size")

	def reset(self):
		if(self.epoch % 20 == 0):
			self.ID2sum= {}
			self.ID2count = {}
		self.epoch += 1

