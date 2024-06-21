import torch
import torch.nn as nn 
import copy
from typing import List, Union, Tuple
from zono_sparse_gen import ZonoSparseGeneration
import numpy as np
import gc





class AbstractReLU(nn.Module):
    max_symbol = np.inf
    recycling = 1

    def __init__(self,max_symbols:Union[int,bool]=False):
        super(AbstractReLU,self).__init__()
   
    @staticmethod
    def abstract_relu(
                      x_center:torch.Tensor,
                      x_abs:torch.Tensor,
                      trash_layer:torch.Tensor,
                      start_index: int = None,
                      add_symbol:bool=False,
                      device:torch.device=torch.device("cpu"))->Tuple[torch.Tensor, torch.Tensor, torch.Tensor ]:


        x_center = x_center.to(device)
        x_abs = x_abs.to(device)
        trash_layer = trash_layer.to(device)
        x_min = x_center-x_abs
        x_max = x_center+x_abs
      
        sgn_min = torch.sign(x_min)
        sgn_max = torch.sign(x_max)
        sgn = sgn_min+sgn_max

        coef_approx_linear = x_max/(torch.abs(x_max)+torch.abs(x_min))
        coef_approx_linear = torch.where(torch.isnan(coef_approx_linear),torch.zeros_like(coef_approx_linear),coef_approx_linear)
        
        bias_approx_linear = x_max*(1-coef_approx_linear)/2
        noise_approx_linear = torch.abs(bias_approx_linear)
      
        
        mask_p = (sgn==0)
    
        mask_1 =(sgn==2) + (sgn==1)
        mask_0 = (sgn==-2)+(sgn==-1)

        mask_center = torch.zeros_like(x_center)
       
       
        x_center[mask_p] = coef_approx_linear[mask_p]*x_center[mask_p] + bias_approx_linear[mask_p]
        x_center[mask_0] = 0
        """ 
        x[0,mask_p]=(coef_approx_linear[mask_p]*x[0,mask_p]+bias_approx_linear[mask_p])
        x[0,mask_1]=x[0,mask_1]
        x[0,mask_0]=0
        """
        mask_epsilon = torch.zeros_like(mask_center)
        mask_epsilon[mask_p]= coef_approx_linear[mask_p]
        mask_epsilon[mask_1]= 1
        """
        x[1:-1,mask_p]=coef_approx_linear[mask_p]*x[1:-1,mask_p]
        x[1:-1,mask_1]=x[1:-1,mask_1]
        x[1:-1,mask_0]=0
        """

        trash_layer[mask_p] = noise_approx_linear[mask_p]+torch.abs(coef_approx_linear[mask_p])*trash_layer[mask_p]
        trash_layer[mask_0] = 0

        """
        x[-1,mask_p]=noise_approx_linear[mask_p]+torch.abs(coef_approx_linear[mask_p])*x[-1,mask_p]
        x[-1,mask_1]=x[-1,mask_1]
        x[-1,mask_0]=0
        """
        new_sparse = None
        trash_layer.to('cpu')
        if add_symbol:
             _, new_sparse = ZonoSparseGeneration(trash_layer,from_trash=True,start_index=start_index).total_zono()
             print(new_sparse)
             
             trash_layer = torch.zeros_like(trash_layer)
        if new_sparse is not None:
            new_sparse.to('cpu')

        return x_center.to('cpu'),trash_layer.to('cpu'), mask_epsilon.to('cpu'), new_sparse
    

def main():
    x,t,m,n = AbstractReLU().abstract_relu(torch.randn(224,512,512),0.01*torch.randn(224,512,512),torch.zeros(224,512,512),start_index=150,add_symbol=True)
    print(x)
    print(t)
    print(m)
    print(n)

if __name__=="__main__":
    main()

    
    