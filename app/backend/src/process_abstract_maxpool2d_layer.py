import torch
import torch.nn as nn
from app.backend.src.util import get_largest_tensor_size
from app.backend.src.process_linear_layer import static_process_linear_layer
from app.backend.src.abstract_relu import AbstractReLU
import copy


def abstract_addition(abstract_domain_1,abstract_domain_2):
    
    assert abstract_domain_1['perfect_domain']==True and abstract_domain_2['perfect_domain']==True , f'You canot add abstract domain while they are not perfect'
    
    zonotope_1 = abstract_domain_1['zonotope']
    center_1 = abstract_domain_1['center']
    trash_1 = abstract_domain_1['trash']
    zonotope_2 = abstract_domain_2['zonotope']
    center_2 = abstract_domain_2['center']
    trash_2 = abstract_domain_2['trash']
    dimask = get_largest_tensor_size(zonotope_1,zonotope_2)

    zonotope_1 = torch.sparse_coo_tensor(zonotope_1.indices(), zonotope_1.values(),size=dimask)
    zonotope_2 = torch.sparse_coo_tensor(zonotope_2.indices(), zonotope_2.values(),size=dimask)
    zonotope_1 = (zonotope_1 + zonotope_2).coalesce()


    sum_1 = torch.sum(torch.abs(zonotope_1),dim=0).unsqueeze(0).to_dense()
    center_1 += center_2
    trash_1 = torch.abs(trash_1)+torch.abs(trash_2)

    abstract_domain_1['zonotope'] = zonotope_1
    abstract_domain_1['center'] = center_1
    abstract_domain_1['sum'] = sum_1
    abstract_domain_1['trash'] = trash_1
    abstract_domain_1['mask'] = torch.ones_like(center_1)    
    abstract_domain_1['perfect_domain']=True

    return abstract_domain_1


def abstract_substraction(abstract_domain_1,abstract_domain_2):
    
    assert abstract_domain_1['perfect_domain']==True and abstract_domain_2['perfect_domain']==True , f'You canot add abstract domain while they are not perfect'
    
    zonotope_1 = abstract_domain_1['zonotope']
    center_1 = abstract_domain_1['center']
    trash_1 = abstract_domain_1['trash']
    zonotope_2 = abstract_domain_2['zonotope']
    center_2 = abstract_domain_2['center']
    trash_2 = abstract_domain_2['trash']
    dimask = get_largest_tensor_size(zonotope_1,zonotope_2)

    zonotope_1 = torch.sparse_coo_tensor(zonotope_1.indices(), zonotope_1.values(),size=dimask)
    zonotope_2 = torch.sparse_coo_tensor(zonotope_2.indices(), zonotope_2.values(),size=dimask)
    zonotope_1 = (zonotope_1 - zonotope_2).coalesce()


    sum_1 = torch.sum(torch.abs(zonotope_1),dim=0).unsqueeze(0).to_dense()
    center_1 -= center_2
    trash_1 = torch.abs(trash_1)+torch.abs(trash_2)

    abstract_domain_1['zonotope'] = zonotope_1
    abstract_domain_1['center'] = center_1
    abstract_domain_1['sum'] = sum_1
    abstract_domain_1['trash'] = trash_1
    abstract_domain_1['mask'] = torch.ones_like(center_1)    
    abstract_domain_1['perfect_domain']=True

    return abstract_domain_1






def resize_dim_1(abstract_domain_1,abstract_domain):
    """
    Resize the zonotope tensor within the abstract domain.
    
    Parameters:
    abstract_domain_1 (dict): The abstract domain containing the zonotope_1 tensor.
    abstract_domain (dict): The abstract domain containing the zonotope tensor.
    
    Returns:
    dict: The updated abstract domain with the coalesced zonotope tensor.
    """
    dim_zono = abstract_domain_1['zonotope'].size(0)
    zonotope_size = (dim_zono, *abstract_domain['zonotope'].size()[1:])
    
    abstract_domain['zonotope'] = torch.sparse_coo_tensor(abstract_domain['zonotope'].indices(),abstract_domain['zonotope'].values(), size=zonotope_size).coalesce()
    
    


def process_max_pool2D(abstract_domain, maxpool_layer,num_workers,available_ram,device, add_symbol):
    with torch.no_grad():

        ident = nn.Identity()
        if abstract_domain['perfect_domain']==False:
           
            abstract_domain = static_process_linear_layer(abstract_domain,ident,ident,ident,num_workers,available_ram,device,add_symbol)
       
        center = abstract_domain['center']
 
  
        dim_x = center.size(1)
        kernel_size = maxpool_layer.kernel_size
        stride = maxpool_layer.stride
        padding = maxpool_layer.padding
        
        assert kernel_size==2 or kernel_size==[2,2] ,f"Maxpool2D kernel size {kernel_size}. A kernel size different of 2 is not supported"

        kernel_size = 2


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

        ident = nn.Identity()

        

        abstract_domain_1 = static_process_linear_layer(copy.deepcopy(abstract_domain),
                                                        conv_0,
                                                        conv_0,
                                                        conv_0,
                                                        num_workers,
                                                        available_ram,
                                                        device,
                                                        add_symbol)

       
        abstract_domain_1= AbstractReLU.evaluate(abstract_domain_1)
        
        abstract_domain_1 = static_process_linear_layer(abstract_domain_1,
                                                            ident,                            
                                                            ident,
                                                            ident,
                                                            num_workers,
                                                            available_ram,
                                                            device,
                                                            add_symbol)
        
        resize_dim_1(abstract_domain_1,abstract_domain)
        abstract_domain_2 = static_process_linear_layer(copy.deepcopy(abstract_domain),
                                                        conv_1,
                                                        conv_1,
                                                        conv_1,
                                                        num_workers,
                                                        available_ram,
                                                        device,
                                                        add_symbol)
        
        
        abstract_domain_1 = abstract_addition(abstract_domain_1,abstract_domain_2)

        
        abstract_domain_2= static_process_linear_layer(copy.deepcopy(abstract_domain),
                                                        conv_2,
                                                        conv_2,
                                                        conv_2,
                                                        num_workers,
                                                        available_ram,
                                                        device,
                                                        add_symbol)
        
        
  
        abstract_domain_2 = abstract_substraction(abstract_domain_2,abstract_domain_1)
        
       
        abstract_domain_3 = AbstractReLU.evaluate(abstract_domain_2)

        abstract_domain_3 = static_process_linear_layer(abstract_domain_3,
                                                        ident,
                                                        ident,
                                                        ident,
                                                        num_workers,
                                                        available_ram,
                                                        device,
                                                        add_symbol)
        resize_dim_1(abstract_domain_3,abstract_domain)
        
        abstract_domain_3 = abstract_addition(abstract_domain_1,abstract_domain_3)
        
        

        abstract_domain_2 = static_process_linear_layer(copy.deepcopy(abstract_domain),
                                                        conv_3,
                                                        conv_3,
                                                        conv_3, 
                                                        num_workers,
                                                        available_ram,
                                                        device,
                                                        add_symbol)
        abstract_domain_2 = abstract_substraction(abstract_domain_2,abstract_domain_3)

       
        abstract_domain_4= AbstractReLU.evaluate(abstract_domain_2)

        abstract_domain_4= static_process_linear_layer(abstract_domain_4,
                                                        ident, 
                                                        ident, 
                                                        ident, 
                                                        num_workers,
                                                        available_ram, 
                                                        device, 
                                                        add_symbol)
        abstract_domain_4 = abstract_addition(abstract_domain_3,abstract_domain_4)
       
 
        
        return abstract_domain_4


