import torch
import random
import numpy as np
import torch.nn as nn
import ray
import os

def ensure_ray_initialized():
    """
    Ensure that Ray is initialized with the appropriate number of CPUs.

    If Ray is not already initialized, this function initializes Ray with the number of CPUs available on the system.
    """
    if not ray.is_initialized():
        num_cpus = os.cpu_count()
        print('ray_init')
        ray.init(num_cpus=num_cpus)

class SimpleCNN(nn.Module):
    """
    A simple convolutional neural network (CNN) model.

    This model consists of two convolutional layers followed by max-pooling layers, and two fully connected layers.

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer.
        relu1 (nn.ReLU): ReLU activation for the first convolutional layer.
        pool1 (nn.MaxPool2d): Max-pooling layer for the first convolutional layer.
        conv2 (nn.Conv2d): Second convolutional layer.
        relu2 (nn.ReLU): ReLU activation for the second convolutional layer.
        pool2 (nn.MaxPool2d): Max-pooling layer for the second convolutional layer.
        flatten (nn.Flatten): Flatten layer to reshape the tensor for the fully connected layers.
        fc1 (nn.Linear): First fully connected layer.
        relu3 (nn.ReLU): ReLU activation for the first fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        relu4 (nn.ReLU): ReLU activation for the second fully connected layer.
    """
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
        """
        Define the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        return x

def sparse_tensor_stats(sparse_tensor):
    """
    Calculate statistics for a sparse tensor.

    Args:
        sparse_tensor (torch.sparse.FloatTensor): The input sparse tensor.

    Returns:
        tuple: A tuple containing the percentage of non-zero elements and the memory gain percentage.

    Raises:
        ValueError: If the tensor is not in COO format.
    """
    if sparse_tensor.layout != torch.sparse_coo:
        raise ValueError("The tensor must be in COO format")
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
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.deterministic = True
    torch.backends.cuda.benchmark = False

def resize_sparse_coo_tensor(sparse_tensor, new_size):
    """
    Resize a sparse COO tensor to a new size.

    Args:
        sparse_tensor (torch.sparse.FloatTensor): The input sparse tensor.
        new_size (tuple): The new size for the sparse tensor.

    Returns:
        torch.sparse.FloatTensor: The resized sparse tensor.
    """
    indices = sparse_tensor.coalesce().indices()
    values = sparse_tensor.coalesce().values()
    mask = torch.all(indices < torch.tensor(new_size).unsqueeze(1), dim=0)
    new_indices = indices[:, mask]
    new_values = values[mask]
    new_sparse_tensor = torch.sparse_coo_tensor(new_indices, new_values, new_size)
    return new_sparse_tensor.coalesce()

def get_largest_tensor_size(tensor1, tensor2):
    """
    Get the size of the tensor with the largest first dimension.

    Args:
        tensor1 (torch.sparse.FloatTensor): The first sparse tensor.
        tensor2 (torch.sparse.FloatTensor): The second sparse tensor.

    Returns:
        torch.Size: The size of the tensor with the largest first dimension.
    """
    size1 = tensor1.shape[0]
    size2 = tensor2.shape[0]
    return tensor1.shape if size1 > size2 else tensor2.shape

def sparse_dense_broadcast_mult(sparse_tensor, multiplicative_tensor):
    """
    Multiply the values of a sparse tensor by the corresponding values of a dense tensor.

    Args:
        sparse_tensor (torch.sparse.FloatTensor): The input sparse tensor.
        multiplicative_tensor (torch.Tensor): The dense tensor to multiply with.

    Returns:
        torch.sparse.FloatTensor: The resulting sparse tensor after multiplication.
    """
    assert sparse_tensor.shape[1:] == multiplicative_tensor.shape[1:], "The dimensions of the tensors must be compatible"
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
