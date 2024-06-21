import torch
import torch.nn as nn 
import copy
from typing import List, Union, Tuple
from zono_sparse_gen import ZonoSparseGeneration
from abstract_relu import AbstractReLU
import numpy as np
import gc


class AbstractMaxpool2D(nn.Module):
    max_symbol = np.inf
    recycling = 1
    
    def __init__(self,max_symbols:Union[int,bool]=False):
       super(AbstractMaxpool2D,self).__init__()
    
 
    @staticmethod
    def abstract_maxpool2D(maxpool:nn.Module,
                            x_center:torch.Tensor,
                            x_abs:torch.Tensor,
                            trash_layer:torch.Tensor,
                            start_index: int = None,
                            add_symbol:bool=False,
                            device:torch.device=torch.device("cpu"))->Tuple[torch.Tensor, torch.Tensor, torch.Tensor ]:
        
        maxpool = maxpool.to(device)
        kernel_size = maxpool.kernel_size
        
        assert kernel_size==2,f"Maxpool2D kernel size {kernel_size}. A kernel size different of 2 is not supported"
        
        stride = maxpool.stride
        padding = maxpool.padding

        dim_x =len(x_center)
        x_min = x_center-x_abs
        x_max = x_center+x_abs
        

        conv_0 = nn.Conv2d(dim_x, dim_x, kernel_size=kernel_size, stride=stride, padding=padding,groups=dim_x)
        conv_1 = nn.Conv2d(dim_x, dim_x, kernel_size=kernel_size, stride=stride, padding=padding,groups=dim_x)
        conv_2 = nn.Conv2d(dim_x, dim_x, kernel_size=kernel_size, stride=stride, padding=padding,groups=dim_x)
        conv_3 = nn.Conv2d(dim_x, dim_x, kernel_size=kernel_size, stride=stride, padding=padding,groups=dim_x)
        
        w_0 = torch.tensor([[[[1., -1.], [0, 0.]]]])
        w_1 = torch.tensor([[[[0., 1.], [0, 0.]]]])
        w_2 = torch.tensor([[[[0., 0.], [0., 1.]]]])
        w_3 = torch.tensor([[[[0., 0.], [1., 0.]]]])
        
        w_0 = w_0.expand(dim_x,-1,-1,-1)
        w_1 = w_1.expand(dim_x,-1,-1,-1)
        w_2 = w_2.expand(dim_x,-1,-1,-1)
        w_3 = w_3.expand(dim_x,-1,-1,-1)
        
        conv_0.weight.data = w_0
        conv_0.bias.data =  torch.zeros(dim_x)
        conv_1.weight.data = w_1
        conv_1.bias.data =  torch.zeros(dim_x)
        conv_2.weight.data = w_2
        conv_2.bias.data =  torch.zeros(dim_x)
        conv_3.weight.data = w_3
        conv_3.bias.data =  torch.zeros(dim_x)
       #max(a,b,c,d) = relu(relu(relu(a-b)+b)+c)+d)






        
        x_result,x_min_result,x_max_result,x_true_result  = AbstractLinear.abstract_conv2D(conv_0,x,x_true,device=device)
        x_result,x_min_result,x_max_result,x_true_result = AbstractReLU.abstract_relu_conv2D(x_result,x_min_result,x_max_result,x_true_result,add_symbol=add_symbol,device=device)
        x_result_1,x_min_result_1,x_max_result_1,x_true_result_1  = AbstractLinear.abstract_conv2D(conv_1,x,x_true,device=device)
        
        x_result = AbstractBasic.abstract_addition(x_result_1,x_result)
        
        x_min_result += x_min_result_1
        x_max_result += x_max_result_1
        x_true_result += x_true_result_1
        x_result_1,x_min_result_1,x_max_result_1,x_true_result_1  = AbstractLinear.abstract_conv2D(conv_2,x,x_true,device=device)
        
        x_result_1 = AbstractBasic.abstract_substraction(x_result_1, x_result)
        
        x_min_result_1 -= x_min_result
        x_max_result_1 -= x_max_result
        x_true_result_1 -= x_true_result
        x_result_2,x_min_result_2,x_max_result_2,x_true_result_2  = AbstractReLU.abstract_relu_conv2D(x_result_1,x_min_result_1,x_max_result_1,x_true_result_1,add_symbol=add_symbol,device=device)
        
        x_result_2 = AbstractBasic.abstract_addition(x_result_2, x_result)
        
        x_min_result_2 += x_min_result
        x_max_result_2 += x_max_result
        x_true_result_2 += x_true_result
        x_result_1,x_min_result_1,x_max_result_1,x_true_result_1  = AbstractLinear.abstract_conv2D(conv_3,x,x_true,device=device)
        
        x_result_1 =AbstractBasic.abstract_substraction(x_result_1, x_result_2)
        
        x_min_result_1 -= x_min_result_2
        x_max_result_1 -= x_max_result_2
        x_true_result_1 -= x_true_result_2
        x_result_3,x_min_result_3,x_max_result_3,x_true_result_3  = AbstractReLU.abstract_relu_conv2D(x_result_1,x_min_result_1,x_max_result_1,x_true_result_1,add_symbol=add_symbol,device=device)
        
        x_result_3 = AbstractBasic.abstract_addition(x_result_3,x_result_2)
        
        x_min_result_3 += x_min_result_2
        x_max_result_3 += x_max_result_2
        x_true_result_3 += x_true_result_2
        x = x_result_3
        
        x_min = x_min_result_3
        x_max = x_max_result_3
        
        x_true = x_final
        
       
        return x,x_min,x_max,x_true
