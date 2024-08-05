import torch 
import timeit
import torch.nn as nn 
import time


class TestCNN(nn.Module):
    def __init__(self):
        super(TestCNN,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3)
        self.conv1.bias.zero_()
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3)
        self.conv2.bias.zero_()
       # self.conv3 = nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=3)
        #self.conv3.bias.zero_()
    def forward_1(self,x):
        x= self.conv1(x)
        x=self.conv2(x)
        #x= self.conv3(x)
        return x
    
    def forward(self,x):
        x= self.conv1(x)
        return(x)
    def combine_convs(self):
        weight1 = self.conv1.weight.data
        weight2 = self.conv2.weight.data
        weight3 = weight2.flatten().t()@weight1.flatten()

        combined_weight = nn.Conv2d(weight1.unsqueeze(0), weight2, stride=1, padding=1,kernel_size=1).squeeze(0)
        return combined_weight






test = torch.randn(3,1024,1024)
with torch.no_grad():
    model =TestCNN()

    time_start = time.time()
    for i in range(1):
        model(test)
    time_stop = time.time()
    time1 =time_stop-time_start
    print(time1)
    time_start = time.time()
    for i in range(1):
        model.forward_1(test)
    time_stop = time.time()
    time1 =time_stop-time_start
    print(time1)


    model= model.combine_convs()
