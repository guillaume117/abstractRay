import torch
import torch.nn as nn
from AbstractRay.backend.src.util import get_largest_tensor_size
from AbstractRay.backend.src.process_linear_layer import static_process_linear_layer,static_process_linear_layer_parrallel
from AbstractRay.backend.src.abstract_relu import AbstractReLU
import copy

def abstract_addition(abstract_domain_1, abstract_domain_2,evaluator_rel=None):
    """
    Add two abstract domains.

    Args:
        abstract_domain_1 (dict): The first abstract domain.
        abstract_domain_2 (dict): The second abstract domain.

    Returns:
        dict: The updated abstract domain after addition.

    Raises:
        AssertionError: If either abstract domain is not perfect.
    """
    assert abstract_domain_1['perfect_domain'] == True and abstract_domain_2['perfect_domain'] == True, \
        'You cannot add abstract domain while they are not perfect'

    zonotope_1 = abstract_domain_1['zonotope']
    center_1 = abstract_domain_1['center']
    trash_1 = abstract_domain_1['trash']
    zonotope_2 = abstract_domain_2['zonotope']
    center_2 = abstract_domain_2['center']
    trash_2 = abstract_domain_2['trash']
    dimask = get_largest_tensor_size(zonotope_1, zonotope_2)

    zonotope_1 = torch.sparse_coo_tensor(zonotope_1.indices(), zonotope_1.values(), size=dimask)
    zonotope_2 = torch.sparse_coo_tensor(zonotope_2.indices(), zonotope_2.values(), size=dimask)
    zonotope_1 = (zonotope_1 + zonotope_2).coalesce()

    sum_1 = torch.sum(torch.abs(zonotope_1), dim=0).unsqueeze(0).to_dense()
    center_1 += center_2
    trash_1 = torch.abs(trash_1) + torch.abs(trash_2)

    abstract_domain_1['zonotope'] = zonotope_1
    abstract_domain_1['center'] = center_1
    abstract_domain_1['sum'] = sum_1
    abstract_domain_1['trash'] = trash_1
    abstract_domain_1['mask'] = torch.ones_like(center_1)
    abstract_domain_1['perfect_domain'] = True

    return abstract_domain_1

def abstract_substraction(abstract_domain_1, abstract_domain_2):
    """
    Subtract one abstract domain from another.

   AbstractRay
    Args:
        abstract_domain_1 (dict): The first abstract domain.
        abstract_domain_2 (dict): The second abstract domain to subtract.

    Returns:
        dict: The updated abstract domain after subtraction.

    Raises:
        AssertionError: If either abstract domain is not perfect.
    """
    assert abstract_domain_1['perfect_domain'] == True and abstract_domain_2['perfect_domain'] == True, \
        'You cannot subtract abstract domain while they are not perfect'

    zonotope_1 = abstract_domain_1['zonotope']
    center_1 = abstract_domain_1['center']
    trash_1 = abstract_domain_1['trash']
    zonotope_2 = abstract_domain_2['zonotope']
    center_2 = abstract_domain_2['center']
    trash_2 = abstract_domain_2['trash']
    dimask = get_largest_tensor_size(zonotope_1, zonotope_2)

    zonotope_1 = torch.sparse_coo_tensor(zonotope_1.indices(), zonotope_1.values(), size=dimask)
    zonotope_2 = torch.sparse_coo_tensor(zonotope_2.indices(), zonotope_2.values(), size=dimask)
    zonotope_1 = (zonotope_1 - zonotope_2).coalesce()

    sum_1 = torch.sum(torch.abs(zonotope_1), dim=0).unsqueeze(0).to_dense()
    center_1 -= center_2
    trash_1 = torch.abs(trash_1) + torch.abs(trash_2)

    abstract_domain_1['zonotope'] = zonotope_1
    abstract_domain_1['center'] = center_1
    abstract_domain_1['sum'] = sum_1
    abstract_domain_1['trash'] = trash_1
    abstract_domain_1['mask'] = torch.ones_like(center_1)
    abstract_domain_1['perfect_domain'] = True

    return abstract_domain_1

def resize_dim_1(abstract_domain_1, abstract_domain):
    """
    Resize the zonotope tensor within the abstract domain.

    Args:
        abstract_domain_1 (dict): The abstract domain containing the zonotope_1 tensor.
        abstract_domain (dict): The abstract domain containing the zonotope tensor.

    Returns:
        dict: The updated abstract domain with the coalesced zonotope tensor.
    """
    dim_zono = abstract_domain_1['zonotope'].size(0)
    zonotope_size = (dim_zono, *abstract_domain['zonotope'].size()[1:])

    abstract_domain['zonotope'] = torch.sparse_coo_tensor(abstract_domain['zonotope'].indices(), 
                                                          abstract_domain['zonotope'].values(), 
                                                          size=zonotope_size).coalesce()

def process_max_pool2D(abstract_domain, evaluator_rel, maxpool_layer, num_workers, available_ram, device, add_symbol):

    """
    Process a MaxPool2D layer within the abstract domain.

    Args:
        abstract_domain (dict): The abstract domain to process.
        maxpool_layer (nn.MaxPool2d): The MaxPool2D layer.
        num_workers (int): Number of workers for processing.
        available_ram (int): Available RAM in GB.
        device (torch.device): The device to run the processing on.
        add_symbol (bool): Flag to indicate whether to add a symbol.

    Returns:
        dict: The updated abstract domain after processing the MaxPool2D layer.

    Raises:
        AssertionError: If the kernel size of the MaxPool2D layer is not 2.
    """
    with torch.no_grad():
        ident = nn.Identity()
        if not abstract_domain['perfect_domain']:
            abstract_domain = static_process_linear_layer_parrallel(evaluator_rel=evaluator_rel,abstract_domain=abstract_domain, function_center=ident, function_epsilon=ident, function_trash=ident, 
                                                          num_workers=num_workers, available_RAM=available_ram, device=device, add_symbol=add_symbol)
   
        center = abstract_domain['center']
        dim_x = center.size(1)
        kernel_size = maxpool_layer.kernel_size
        stride = maxpool_layer.stride
        padding = maxpool_layer.padding

        assert kernel_size == 2 or kernel_size == [2, 2], \
            f"Maxpool2D kernel size {kernel_size}. A kernel size different from 2 is not supported"

        conv_0 = nn.Conv2d(dim_x, dim_x, kernel_size=2, stride=stride, padding=padding, groups=dim_x)
        conv_1 = nn.Conv2d(dim_x, dim_x, kernel_size=2, stride=stride, padding=padding, groups=dim_x)
        conv_2 = nn.Conv2d(dim_x, dim_x, kernel_size=2, stride=stride, padding=padding, groups=dim_x)
        conv_3 = nn.Conv2d(dim_x, dim_x, kernel_size=2, stride=stride, padding=padding, groups=dim_x)

        w_0 = torch.tensor([[[[1., -1.], [0, 0.]]]]).expand(dim_x, -1, -1, -1)
        w_1 = torch.tensor([[[[0., 1.], [0, 0.]]]]).expand(dim_x, -1, -1, -1)
        w_2 = torch.tensor([[[[0., 0.], [0., 1.]]]]).expand(dim_x, -1, -1, -1)
        w_3 = torch.tensor([[[[0., 0.], [1., 0.]]]]).expand(dim_x, -1, -1, -1)

        conv_0.weight.data = w_0
        conv_0.bias.data = torch.zeros(dim_x)
        conv_1.weight.data = w_1
        conv_1.bias.data = torch.zeros(dim_x)
        conv_2.weight.data = w_2
        conv_2.bias.data = torch.zeros(dim_x)
        conv_3.weight.data = w_3
        conv_3.bias.data = torch.zeros(dim_x)

        ident = nn.Identity()

        evaluator_rel.copy(indice=1)
        abstract_domain_1 = static_process_linear_layer_parrallel(evaluator_rel,copy.deepcopy(abstract_domain),
                                                        conv_0, conv_0, conv_0,
                                                        num_workers, available_ram, device, add_symbol,over_copy=True,indice_copy=1)
      

        abstract_domain_1 = AbstractReLU.evaluate(abstract_domain_1)


        abstract_domain_1 = static_process_linear_layer_parrallel(evaluator_rel,abstract_domain_1,
                                                        ident, ident, ident,
                                                        num_workers, available_ram, device, add_symbol,over_copy=True,indice_copy=1)
        
   
        resize_dim_1(abstract_domain_1, abstract_domain)
     

        evaluator_rel.copy(indice=2)
        abstract_domain_2 = static_process_linear_layer_parrallel(evaluator_rel,copy.deepcopy(abstract_domain),
                                                        conv_1, conv_1, conv_1,
                                                        num_workers, available_ram, device, add_symbol,over_copy=True, indice_copy=2)
        

        abstract_domain_1 = abstract_addition(abstract_domain_1, abstract_domain_2)


        evaluator_rel.add(indice_1=1,indice_2=2,result_indice=1)

        
        evaluator_rel.copy(indice=2)
        abstract_domain_2 = static_process_linear_layer_parrallel(evaluator_rel,copy.deepcopy(abstract_domain),
                                                        conv_2, conv_2, conv_2,
                                                        num_workers, available_ram, device, add_symbol,over_copy=True,indice_copy=2)
    

        abstract_domain_2 = abstract_substraction(abstract_domain_2, abstract_domain_1)
        evaluator_rel.sub(indice_1=2,indice_2=1,result_indice=2)

        abstract_domain_3 = AbstractReLU.evaluate(abstract_domain_2)
        evaluator_rel.copy(indice=3,indice_in=2)

        abstract_domain_3 = static_process_linear_layer_parrallel(evaluator_rel,abstract_domain_3,
                                                        ident, ident, ident,
                                                        num_workers, available_ram, device, add_symbol,over_copy=True,indice_copy=2)
        resize_dim_1(abstract_domain_3, abstract_domain)

        evaluator_rel.add(indice_1=1,indice_2=3,result_indice=3)

        abstract_domain_3 = abstract_addition(abstract_domain_1, abstract_domain_3)

        evaluator_rel.copy(indice=2)
        abstract_domain_2 = static_process_linear_layer_parrallel(evaluator_rel,copy.deepcopy(abstract_domain),
                                                        conv_3, conv_3, conv_3,
                                                        num_workers, available_ram, device, add_symbol,over_copy=True,indice_copy=2)
        evaluator_rel.sub(indice_1=2,indice_2=3,result_indice=2)
        abstract_domain_2 = abstract_substraction(abstract_domain_2, abstract_domain_3)

        abstract_domain_2 = AbstractReLU.evaluate(abstract_domain_2)



        abstract_domain_2 = static_process_linear_layer_parrallel(evaluator_rel,abstract_domain_2,
                                                        ident, ident, ident,
                                                        num_workers, available_ram, device, add_symbol,over_copy=True,indice_copy=2)

        abstract_domain = abstract_addition(abstract_domain_3, abstract_domain_2)
        evaluator_rel.add(indice_1=3,indice_2=2)


        return abstract_domain
