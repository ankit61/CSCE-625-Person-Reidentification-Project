import sys
import torch

model_path = sys.argv[1]
checkpoint = torch.load(model_path)
best_loss = checkpoint['best_loss']
model = CAE()
model.cuda()
model.load_state_dict(checkpoint['state_dict'])

