import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from RealConv import RealConv2d

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms

torch.set_default_tensor_type(torch.cuda.FloatTensor)

BATCH_SIZE = 64
EPOCHS = 300
LEARNING_RATE = .05
WEIGHT_DECAY = 5e-4
MOMENTUM = .9
NUM_CLASSES = 10

class VGGNet(nn.Module):

    def __init__(self, device):
        super(VGGNet, self).__init__()
        
        self.conv_64_1 = RealConv2d(3, 64, kernel_size=3, padding=1).to(device)
        
        self.conv_128_1 = RealConv2d(64, 128, kernel_size=3, padding=1).to(device)
        
        self.conv_256_1 = RealConv2d(128, 256, kernel_size=3, padding=1).to(device)
        self.conv_256_2 = RealConv2d(256, 256, kernel_size=3, padding=1).to(device)
        
        self.conv_512_1 = RealConv2d(256, 512, kernel_size=3, padding=1).to(device)
        self.conv_512_2 = RealConv2d(512, 512, kernel_size=3, padding=1).to(device)
        
        self.conv_512_4 = RealConv2d(512, 512, kernel_size=3, padding=1).to(device)
        self.conv_512_5 = RealConv2d(512, 512, kernel_size=3, padding=1).to(device)

        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, NUM_CLASSES),
        )

        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        
        x = F.max_pool2d(self.conv_64_1(x), kernel_size=2, stride=2)
        
        x = F.max_pool2d(self.conv_128_1(x), kernel_size=2, stride=2)

        x = F.max_pool2d(self.conv_256_2(self.conv_256_1(x)), kernel_size=2, stride=2)

        x = F.max_pool2d(self.conv_512_2(self.conv_512_1(x)), kernel_size=2, stride=2)

        x = F.max_pool2d(self.conv_512_5(self.conv_512_4(x)), kernel_size=2, stride=2)

        x = x.view(x.size(0), -1)

        x = self.classifier(x)

        return x

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
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        
        lossfn = nn.CrossEntropyLoss().cuda()
        loss = lossfn(output, target)
        
        loss.backward()
        optimizer.step()
        
        if batch_idx % 5 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    lr = LEARNING_RATE * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    device = torch.device("cuda")
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../CIFAR', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.RandomHorizontalFlip(),
                           transforms.RandomCrop(32, 4),
                           transforms.ToTensor(),
                           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                       ])),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../CIFAR', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                       ])),
        batch_size=BATCH_SIZE, shuffle=True,num_workers=4, pin_memory=True)

    print('Conv Net Architecture: ')
    net = VGGNet(device)
    print(net)
    model = net.to(device)
    optimizer = optim.SGD(model.parameters(), LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    for epoch in range(1, EPOCHS + 1):
        adjust_learning_rate(optimizer, epoch)
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

main()
