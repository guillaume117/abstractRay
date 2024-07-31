import torch
import torch.nn as nn
from AbstractRay.backend.src.sparse_evaluation import SparseEvaluation
from AbstractRay.backend.src.zono_sparse_gen import ZonoSparseGeneration

def static_dim_chunk(input, available_RAM):
    """
    Calculate the chunk size for processing based on available RAM.

    Args:
        input (torch.Tensor): The input tensor.
        available_RAM (int): The available RAM in GB.

    Returns:
        int: The calculated chunk size.
    """
    dense_memory_footprint = torch.prod(torch.tensor(input.shape)) * 4 / 1e9
    return int(max(1, available_RAM // (9 * dense_memory_footprint)))

def static_process_center_layer(input, function):
    """
    Process the center layer with the given function.

    Args:
        input (torch.Tensor): The input tensor.
        function (nn.Module): The function to process the tensor.

    Returns:
        torch.Tensor: The output tensor after processing.
    """
    with torch.no_grad():
        return function(input)

def static_process_trash_layer(input, function):
    """
    Process the trash layer with the given function.

    Args:
        input (torch.Tensor): The input tensor.
        function (nn.Module): The function to process the tensor.

    Returns:
        torch.Tensor: The output tensor after processing.
    """
    with torch.no_grad():
        function = function.to('cpu')
        input = input.to('cpu')
        return function(input)

def static_process_linear_layer(abstract_domain, function_center, function_epsilon, function_trash, num_workers, available_RAM, device, add_symbol=True):
    """
    Process a linear layer within the abstract domain.

    Args:
        abstract_domain (dict): The abstract domain to process.
        function_center (nn.Module): The function to process the center tensor.
        function_epsilon (nn.Module): The function to process the epsilon tensor.
        function_trash (nn.Module): The function to process the trash tensor.
        num_workers (int): Number of workers for processing.
        available_RAM (int): Available RAM in GB.
        device (torch.device): The device to run the processing on.
        add_symbol (bool, optional): Flag to indicate whether to add a symbol. Defaults to True.

    Returns:
        dict: The updated abstract domain after processing the linear layer.
    """
    zonotope = abstract_domain['zonotope']
    center = abstract_domain['center']
    sum_abs = abstract_domain['sum']
    trash = abstract_domain['trash']
    mask_epsilon = abstract_domain['mask']

    with torch.no_grad():
        dim_chunk_val_input = static_dim_chunk(center, available_RAM)
        center = static_process_center_layer(center, function_center)
        sum_abs = torch.zeros_like(center)
        dim_chunk_val_output = static_dim_chunk(center, available_RAM)
        dim_chunk_val = min(dim_chunk_val_input, dim_chunk_val_output)

        evaluator = SparseEvaluation(
                                    zonotope,
                                    chunk_size=dim_chunk_val,
                                    function=function_epsilon,
                                    mask_coef=mask_epsilon,
                                    device=device
                                    )

        zonotope = evaluator.evaluate_all_chunks(num_workers=num_workers)
        len_zono = zonotope.size(0)
        mask_epsilon = torch.ones_like(mask_epsilon)

        if add_symbol and not torch.equal(trash, torch.zeros_like(trash)):
            new_sparse = ZonoSparseGeneration().zono_from_tensor(trash, start_index=len_zono).coalesce()
            if new_sparse is not None:
                evaluator_new_noise = SparseEvaluation(
                                                        new_sparse,
                                                        chunk_size=dim_chunk_val,
                                                        function=function_epsilon,
                                                        mask_coef=mask_epsilon,
                                                        device=device
                                                        )
                new_sparse = evaluator_new_noise.evaluate_all_chunks(num_workers=num_workers)
                zonotope = torch.sparse_coo_tensor(zonotope.indices(), zonotope.values(), size=new_sparse.size()).coalesce()
                zonotope += new_sparse
                trash = static_process_trash_layer(trash, function_trash)
                trash = torch.zeros_like(trash)
        else:
            trash = static_process_trash_layer(trash, function_trash)

        sum_abs = torch.sum(torch.abs(zonotope), dim=0).unsqueeze(0).to_dense() + torch.abs(trash)

        abstract_domain['zonotope'] = zonotope
        abstract_domain['center'] = center
        abstract_domain['sum'] = sum_abs
        abstract_domain['trash'] = trash
        abstract_domain['mask'] = torch.ones_like(center)
        abstract_domain['perfect_domain'] = True
  

        return abstract_domain



def static_process_linear_layer_parrallel(evaluator_rel, abstract_domain, function_center, function_epsilon, function_trash, num_workers, available_RAM, device, add_symbol=False,over_copy=False,indice_copy=None):
    """
    Process a linear layer within the abstract domain.

    Args:
        abstract_domain (dict): The abstract domain to process.
        function_center (nn.Module): The function to process the center tensor.
        function_epsilon (nn.Module): The function to process the epsilon tensor.
        function_trash (nn.Module): The function to process the trash tensor.
        num_workers (int): Number of workers for processing.
        available_RAM (int): Available RAM in GB.
        device (torch.device): The device to run the processing on.
        add_symbol (bool, optional): Flag to indicate whether to add a symbol. Defaults to True.

    Returns:
        dict: The updated abstract domain after processing the linear layer.
    """
    zonotope = abstract_domain['zonotope']
   
    center = abstract_domain['center']
    sum_abs = abstract_domain['sum']
    trash = abstract_domain['trash']
    mask_epsilon = abstract_domain['mask']

    with torch.no_grad():
        dim_chunk_val_input = static_dim_chunk(center, available_RAM)
        center = static_process_center_layer(center, function_center)
        sum_abs = torch.zeros_like(center)
        dim_chunk_val_output = static_dim_chunk(center, available_RAM)
        dim_chunk_val = min(dim_chunk_val_input, dim_chunk_val_output)
   

        if zonotope is not None:
            len_zono = zonotope.size(0)
            evaluator = SparseEvaluation(
                                        zonotope,
                                        chunk_size=dim_chunk_val,
                                        function=function_epsilon,
                                        mask_coef=mask_epsilon,
                                        device=device
                                        )

            zonotope = evaluator.evaluate_all_chunks(num_workers=num_workers)
        else: 
            len_zono= 0
        sum_epsilon_rel = evaluator_rel.forward(function=function_epsilon,
                                                    mask=mask_epsilon,
                                                    chunk_size=dim_chunk_val,
                                                    over_copy=over_copy,
                                                    indice_copy=indice_copy)
        
        
         
        
        mask_epsilon = torch.ones_like(mask_epsilon)
     
        if add_symbol and not torch.equal(trash, torch.zeros_like(trash)):
           
            new_sparse = ZonoSparseGeneration().zono_from_tensor(trash, start_index=len_zono).coalesce()
            if new_sparse is not None:
    
                evaluator_new_noise = SparseEvaluation(
                                                        new_sparse,
                                                        chunk_size=dim_chunk_val,
                                                        function=function_epsilon,
                                                        mask_coef=mask_epsilon,
                                                        device=device
                                                        )
                new_sparse = evaluator_new_noise.evaluate_all_chunks(num_workers=num_workers)
                if zonotope is not  None:
                    zonotope = torch.sparse_coo_tensor(zonotope.indices(), zonotope.values(), size=new_sparse.size()).coalesce()
                    zonotope += new_sparse
                else : 
                    zonotope = new_sparse
                    

                trash = static_process_trash_layer(trash, function_trash)
                trash = torch.zeros_like(trash)
    
        else:
            trash = static_process_trash_layer(trash, function_trash)
          
        if zonotope is not None:
            sum_abs = torch.sum(torch.abs(zonotope), dim=0).unsqueeze(0).to_dense() + torch.abs(trash) + sum_epsilon_rel
        else: 
            sum_abs = torch.abs(trash) + sum_epsilon_rel

        abstract_domain['zonotope'] = zonotope
        abstract_domain['center'] = center
        abstract_domain['sum'] = sum_abs
        abstract_domain['trash'] = trash
        abstract_domain['mask'] = torch.ones_like(center)
        abstract_domain['perfect_domain'] = True
       
   
        return abstract_domain
