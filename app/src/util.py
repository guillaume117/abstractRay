
import torch

import random
import numpy as np


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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resize_sparse_coo_tensor(sparse_tensor, new_size):
        """This function aims to resize a sparse tensor to a new_size"""
    
        indices = sparse_tensor.coalesce().indices()
        values = sparse_tensor.coalesce().values()
        
    
        mask = torch.all(indices < torch.tensor(new_size).unsqueeze(1), dim=0)
        new_indices = indices[:, mask]
        new_values = values[mask]

        new_sparse_tensor = torch.sparse_coo_tensor(new_indices, new_values, new_size)
        return new_sparse_tensor.coalesce()


