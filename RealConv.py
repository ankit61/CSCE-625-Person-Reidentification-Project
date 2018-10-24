import torch
from math import floor

class RealConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(RealConv2d, self).__init__()
        self.conv = torch.nn.Conv2d(1, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        
        self.kernelSize = kernel_size
        if type(kernel_size) is not tuple:
            self.kernelSize = (kernel_size, kernel_size)

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.outChannels = out_channels

    def getOutputHeight(self, h):
        return floor(((h + (2 * self.padding) - (self.dilation * (self.kernelSize[0] - 1)) - 1) / self.stride) + 1)

    def getOutputWidth(self, w):
        return floor(((w + (2 * self.padding) - (self.dilation * (self.kernelSize[1] - 1)) - 1) / self.stride) + 1)
    
    def forward(self, x):
        temp = torch.empty(x.size(1), x.size(0), self.outChannels, self.getOutputHeight(x.size(2)), self.getOutputWidth(x.size(3)))
        for i in range(0, x.size(1)): #loop through number of channels
            temp[i, :, :, :, :] = self.conv(x[:, i:i+1, :, :]) #FIXME: copy from GPU to CPU?
        return torch.max(temp, 0)[0]
        
