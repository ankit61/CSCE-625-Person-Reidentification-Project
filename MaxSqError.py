import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter

writer = SummaryWriter("/runs/")

class MaxSqError:
	
	def __init__(self, regConst):
		self.ID2sum = {}
		self.ID2count = {}
		self.regConst = regConst
		self.epoch = 0
	
	def __call__(self, pred, code, target, ID):
		if(pred.size() == target.size() and pred.dim() == 4):
			reconstructionLoss = (pred - target).pow(2).view(pred.size(0), pred.size(1), -1).max(2)[0].mean(0).sum()
			#regularizationLoss = self.regConst * torch.mean(torch.abs(code))#.mean(0).sum()
			
			#compute clustering loss
			clusteringLoss = torch.zeros(1, requires_grad=True)
			for i in range(pred.size(0)):
				curID = int(ID[i])
				if(curID in self.ID2sum):
					mean = self.ID2sum[curID] / self.ID2count[curID]
					clusteringLoss = clusteringLoss + F.mse_loss(code[i], mean) 
					self.ID2count[curID] += 1
					self.ID2sum[curID] += code[i]
				else:
					self.ID2count[curID] = 1
					self.ID2sum[curID] = code[i]

			loss = clusteringLoss + reconstructionLoss #+ regularizationLoss
			
			writer.add_scalar("/runs/cae/clustering", clusteringLoss / loss, self.epoch)
			writer.add_scalar("/runs/cae/reconstruction", reconstructionLoss / loss, self.epoch)
			#writer.add_scalar("/runs/cae/regularization", regularizationLoss / loss, self.epoch)			
			self.epoch += 1

			return loss

		else:
			raise Exception("pred and target must have 4D with same size")

	def reset(self):
		pass
		#self.ID2sum= {}
		#self.ID2count = {}

