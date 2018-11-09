import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from RealConv import RealConv2d

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms

from tensorboardX import SummaryWriter

writer = SummaryWriter('/runs/')


torch.set_default_tensor_type(torch.cuda.FloatTensor)

class Net(nn.Module):

    def __init__(self, device):
        super(Net, self).__init__()
        self.conv1 = RealConv2d(1, 10, kernel_size=5).to(device)
        self.conv2 = RealConv2d(10, 20, kernel_size=5).to(device)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 2)
        x = F.max_pool2d(self.conv2(x), 2)
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    batches = len(train_loader)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            writer.add_text('Text', 'text logged at step:' + str(batch_idx), batch_idx)
            writer.add_scalar('data/total_loss', loss.item(), (batches - 1) * epoch + batch_idx)

def main():
    BATCH_SIZE = 256
    EPOCHS = 300
    LEARNING_RATE = .001

    MOMENTUM = .02

    device = torch.device("cuda")
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('/datasets', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=64, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('/datasets', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=BATCH_SIZE, shuffle=True)

    print('Conv Net Architecture: ')
    net = Net(device)
    print(net)
    model = net.to(device)
    optimizer = optim.SGD(model.parameters(), LEARNING_RATE)
    for epoch in range(1, EPOCHS + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)



main()
