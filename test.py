import torch 
import timeit
import torch.nn as nn 
import time

import torch
import matplotlib.pyplot as plt
from AbstractRay.backend.src.unstack_network import UnStackNetwork

class ZonoSparseGeneration:
    """
    A class for generating sparse zonotopes from tensors and noise levels.

    This class provides methods to create zonotopes based on input tensors, noise levels, and specific regions of interest.

    Attributes:
        global_storage (dict): A dictionary to store global indices and values.
    """
    def __init__(self):
        self.global_storage = {'indices': [], 'values': []}


    def zono_from_tensor(self, noise_intensity, start_index=0):
        """
        Generate a zonotope from a tensor.

        Args:
            noise_intensity (torch.Tensor): The tensor representing noise intensity.
            start_index (int, optional): The starting index for the zonotope. Defaults to 0.

        Returns:
            torch.sparse.FloatTensor: The generated zonotope.
        """
        assert noise_intensity.size(0) == 1, 'First dimension size must be 1'
        noise_intensity = noise_intensity.to_sparse().coalesce()
        noise_intensity.indices()[0] = torch.arange(noise_intensity._nnz()) + start_index
        size = (noise_intensity._nnz() + start_index, *noise_intensity.size()[1:])
        zonotope = torch.sparse_coo_tensor(noise_intensity.indices(), noise_intensity.values(), size=size)
        return zonotope.coalesce()



class DummyCNN(nn.Module):

    def __init__(self):
        super(DummyCNN,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=86528,out_features=10)

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x 
    




class TestCNN(nn.Module):
    def __init__(self):
        super(TestCNN,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3)
        self.conv1.bias.zero_()
        self.p1 = torch.randn(3,28,28)
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3)
        self.conv2.bias.zero_()
        self.p2 = torch.randn(64,26,26)
        self.flatten= nn.Flatten()
        self.fc1= nn.Linear(576,10)
        self.fc1.bias.zero_()

    def forward(self,x):
        x= self.conv1(self.p1*x)

        x=self.conv2(self.p2*x)
        #x=self.flatten(x)
       
       
        #x=self.fc1(x)
        return x
    


class TestNN(nn.Module):
    def __init__(self):
        super(TestNN,self).__init__()
        self.fc1 = nn.Linear(in_features=1000,out_features=110)
        self.fc1.bias.zero_()
        self.p1 = torch.randn(1000)
        self.fc2 = nn.Linear(in_features=110,out_features=100)
        self.fc2.bias.zero_()
        self.p2 = torch.randn(110)
        # self.conv3 = nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=3)
        #self.conv3.bias.zero_()
    
        self.matmul= self.p2*self.fc2.weight.data@(self.p1*self.fc1.weight.data)
        self.absMatmul =torch.abs(self.matmul)
    def forward(self,x):
        x= self.fc1(self.p1*x)
        x=self.fc2(self.p2*x)
        #x= self.conv3(x)
        return x

    def direct_eval(self,x):

        return self.matmul@x
    def abs_eval(self,x):
        return self.absMatmul@x
    
    def get_A(self):
        return self.matmul

with torch.no_grad():
    model = TestNN()
    x = torch.randn(1000)
    time_start = time.time()
    for i in range(100):
        model(x)
    time_stop = time.time()
    time1 =time_stop-time_start
    print(time1)
    time_start = time.time()
    for i in range(100):
        model.direct_eval(x)
    time_stop = time.time()
    time1 =time_stop-time_start
    print(time1)



    
    print(model(x)-model.direct_eval(x))


with torch.no_grad():
    x_vect = ZonoSparseGeneration().zono_from_tensor(x.unsqueeze(0))
    print(x_vect)
    
    print(torch.sum(torch.abs(model(x_vect)),dim=0))

    print(model.abs_eval(torch.abs(x)))
    

with torch.no_grad():

    model =TestCNN()
    def determine_matrix( m, p):
        """
        Détermine la matrice de représentation d'une fonction linéaire f: R^m -> R^p.
        
        Arguments:
        m -- dimension de l'espace de départ.
        p -- dimension de l'espace d'arrivée.
        
        Retourne:
        A -- matrice de représentation de f.
        """
        base_vectors = torch.eye(m)
        A = torch.zeros((p, m))
        for i in range(m):
            A[:, i] = model(base_vectors[:, i].unsqueeze(0).view_as(torch.randn(3,28,28))).flatten()
        return A
    




    # Dimensions
    m = 3*28*28
    p = 128*24*24
    x= torch.randn(3,28,28)


    A = determine_matrix( m, p)
    print("Matrice de représentation A:\n", A)

    print(torch.sum(model(x).flatten()-A@(x.flatten())))

