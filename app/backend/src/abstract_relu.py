import torch
import torch.nn as nn 
from typing import List, Union, Tuple
import numpy as np
import gc





class AbstractReLU(nn.Module):


    def __init__(self,max_symbols:Union[int,bool]=False):
        super(AbstractReLU,self).__init__()
   
    @staticmethod
    def evaluate(abstract_domain,
                      device:torch.device=torch.device("cpu"))->Tuple[torch.Tensor, torch.Tensor, torch.Tensor ]:
     
        zonotope = abstract_domain['zonotope']
        center = abstract_domain['center']
        sum =   abstract_domain['sum']
        trash = abstract_domain['trash']
        mask_epsilon = abstract_domain['mask'] 
        #sum_eval = torch.sum(torch.abs(zonotope),dim=0).to_dense()+torch.abs(trash)

        x_center = center.to(device)
        x_abs = sum.to(device)
        trash_layer = trash.to(device)
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

        mask_center = torch.ones_like(x_center)
       
        mask_center[mask_1] = x_center[mask_1]
        mask_center[mask_p] = coef_approx_linear[mask_p]*x_center[mask_p] + bias_approx_linear[mask_p]
        mask_center[mask_0] = 0
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

        """
        if add_symbol:
             _, new_sparse = ZonoSparseGeneration(trash_layer,from_trash=True,start_index=start_index).total_zono()
             print(new_sparse)
             
             trash_layer = torch.zeros_like(trash_layer)
        """
        abstract_domain['zonotope'] = zonotope
        abstract_domain['center'] = mask_center
        abstract_domain['sum'] = sum
        abstract_domain['trash'] = trash_layer.to('cpu')
        abstract_domain['mask'] = mask_epsilon    
        abstract_domain['perfect_domain']=False

        return abstract_domain
    

    