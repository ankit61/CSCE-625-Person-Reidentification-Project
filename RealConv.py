import torch
from math import floor
import time

torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)

class RealConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(RealConv2d, self).__init__()
        self.conv = torch.nn.Conv2d(1, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        
        torch.nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        if self.conv.bias is not None:
            torch.nn.init.constant_(self.conv.bias, 0)
        
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
        print("temp is cuda: ", temp.is_cuda)
        start = time.time()
        for i in range(0, x.size(1)): #loop through number of channels
            temp[i, :, :, :, :] = self.conv(x[:, i:i+1, :, :]) #FIXME: copy from GPU to CPU?
        print("conv times: ", time() - start)
        start = time.time()
        ans = torch.max(temp, 0)[0]
        print("max time:", ans)
        return ans
        
