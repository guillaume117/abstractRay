import torch
import torch.nn as nn 
import copy
from typing import List, Union, Tuple
from zono_sparse_gen import ZonoSparseGeneration
import numpy as np
import gc
import torch.nn.functional as F





class AbstractMaxPool(nn.Module):
    max_symbol = np.inf
    recycling = 1

    def __init__(self,max_symbols:Union[int,bool]=False):
        super(AbstractMaxPool,self).__init__()
   
    @staticmethod
    def abstract_max_pool_min_max(
                      
                      x_center:torch.Tensor,
                      x_abs:torch.Tensor,
                      trash_layer:torch.Tensor,
                      layer=nn.MaxPool2d(2),
                      start_index: int = None,
                      add_symbol:bool=False,
                      device:torch.device=torch.device("cpu"))->Tuple[torch.Tensor, torch.Tensor, torch.Tensor ]:

        kernel_size = layer.kernel_size[0]
        stride = layer.stride[0]
        padding = layer.padding[0]

        x_center = x_center.to(device)
        x_abs = x_abs.to(device)
        trash_layer = trash_layer.to(device)
        x_min = x_center-x_abs
        x_max = x_center+x_abs
      
        pool_min, indices_pool_min =    F.max_pool2d( x_min, kernel_size, stride, return_indices=True)
        pool_max, indices_pool_max =    F.max_pool2d( x_max, kernel_size, stride, return_indices=True)
        
        
        trash_layer=pool_max-pool_min

        x_center = pool_min+trash_layer/2
        mask_epsilon = torch.ones_like(x_center)
      

        return x_center.to('cpu'),trash_layer.to('cpu'), mask_epsilon.to('cpu')
    

    