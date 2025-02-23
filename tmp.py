import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import torch
import torch.nn as nn

class Test(nn.Module):
    def __init__(self):
        super(Test, self).__init__()
        self.conv1 = nn.Conv2d(48,128,3,1,1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(128,48,3,1,1)

    def forward(self,x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x

if __name__=="__main__":
    x = torch.randn(48,1024,1024).cuda()
    model = Test().cuda()
    y = model(x)

