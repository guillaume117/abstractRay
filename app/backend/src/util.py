import torch
import random
import numpy as np
import torch.nn as nn
import ray
import os

"""
def create_sparse_worker(num_cpus,num_gpus, *args, **kwargs):
    SparseWorkerRemote = ray.remote(num_cpus=num_cpus,num_gpus=num_gpus)(SparseWorker)
    return SparseWorkerRemote.remote(*args, **kwargs)
"""
def ensure_ray_initialized():

    if not ray.is_initialized():
        num_cpus = os.cpu_count()
        print('ray_init')
        ray.init(num_cpus=num_cpus)  
        
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)
        self.relu4 = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)

        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        return x

def sparse_tensor_stats(sparse_tensor):
    """This function aims to monitor the benefits it terms of memory footprint from sparse elements vs dense """

    if sparse_tensor.layout != torch.sparse_coo:
        raise ValueError("Le tenseur doit être au format COO")
    total_elements = torch.prod(torch.tensor(sparse_tensor.size()))
    non_zero_elements = sparse_tensor._nnz()
    non_zero_percentage = (non_zero_elements / total_elements) * 100
    dense_memory = total_elements * sparse_tensor.dtype.itemsize
    values_memory = sparse_tensor.values().numel() * sparse_tensor.values().element_size()
    indices_memory = sparse_tensor.indices().numel() * sparse_tensor.indices().element_size()
    sparse_memory = values_memory + indices_memory
    memory_gain_percentage = ((dense_memory - sparse_memory) / dense_memory) * 100
    return non_zero_percentage, memory_gain_percentage


def set_seed(seed):
    """This function aims to set random to deterministics"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure that all operations are deterministic on GPU (if applicable)
    torch.backends.cuda.deterministic = True
    torch.backends.cuda.benchmark = False


def resize_sparse_coo_tensor(sparse_tensor, new_size):
        """This function aims to resize a sparse tensor to a new_size"""
    
        indices = sparse_tensor.coalesce().indices()
        values = sparse_tensor.coalesce().values()
        
    
        mask = torch.all(indices < torch.tensor(new_size).unsqueeze(1), dim=0)
        new_indices = indices[:, mask]
        new_values = values[mask]

        new_sparse_tensor = torch.sparse_coo_tensor(new_indices, new_values, new_size)
        return new_sparse_tensor.coalesce()


def get_largest_tensor_size(tensor1, tensor2):
    """
    Renvoie la taille (shape) du tenseur sparse COO ayant la plus grande première dimension.

    Args:
    tensor1 (torch.sparse_coo_tensor): Premier tenseur sparse COO.
    tensor2 (torch.sparse_coo_tensor): Deuxième tenseur sparse COO.

    Returns:
    torch.Size: La taille du tenseur avec la plus grande première dimension.
    """
    size1 = tensor1.shape[0]
    size2 = tensor2.shape[0]
    return tensor1.shape if size1 > size2 else tensor2.shape

import torch

def sparse_dense_broadcast_mult(sparse_tensor, multiplicative_tensor):
    """
    Multiplie les valeurs des indices [1:] du tenseur sparse par les valeurs correspondantes du tenseur multiplicatif.
    
    Arguments:
    sparse_tensor -- Tenseur sparse de PyTorch de dimension [B, C, H, W].
    multiplicative_tensor -- Tenseur dense de PyTorch de dimension [1, C, H, W].
    
    Retourne:
    new_sparse_tensor -- Nouveau tenseur sparse avec les valeurs mises à jour.
    """
 
    assert sparse_tensor.shape[1:] == multiplicative_tensor.shape[1:], "Les dimensions des tenseurs doivent être compatibles"
    
 
    sparse_indices = sparse_tensor._indices()
    sparse_values = sparse_tensor._values()
    
  
    new_values = torch.empty_like(sparse_values)
    
   
    for i in range(multiplicative_tensor.size(1)):  
        for j in range(multiplicative_tensor.size(2)):  
            for k in range(multiplicative_tensor.size(3)): 
                mask = (sparse_indices[1] == i) & (sparse_indices[2] == j) & (sparse_indices[3] == k)
                new_values[mask] = sparse_values[mask] * multiplicative_tensor[0, i, j, k]

    new_sparse_tensor = torch.sparse_coo_tensor(sparse_indices, new_values, sparse_tensor.size())
    
    return new_sparse_tensor







