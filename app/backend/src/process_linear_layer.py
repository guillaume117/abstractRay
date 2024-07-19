import torch
import torch.nn as nn

from app.backend.src.sparse_evaluation import SparseEvaluation  
from app.backend.src.zono_sparse_gen import ZonoSparseGeneration




def static_dim_chunk(input,available_RAM):
    dense_memory_footprint = torch.prod(torch.tensor(input.shape)) *4/1e9
    return int(max(1,available_RAM//(3*dense_memory_footprint)))

def static_process_center_layer(input, function):
    with torch.no_grad():
        return function(input)


def static_process_trash_layer(input,function):
    with torch.no_grad():
        function=function.to('cpu')
        input = input.to('cpu')
        return function(input)


def static_process_linear_layer(abstract_domain,function_center,function_epsilon,function_trash,num_workers, available_RAM, device, add_symbol=True):
    
    zonotope = abstract_domain['zonotope']
    center = abstract_domain['center']
    sum =   abstract_domain['sum']
    trash = abstract_domain['trash']
    mask_epsilon = abstract_domain['mask']    

    with torch.no_grad():
        
        dim_chunk_val_input = static_dim_chunk(center, available_RAM)
        center  = static_process_center_layer(center,function_center)
        sum_abs =torch.zeros_like(center)
        dim_chunk_val_output = static_dim_chunk(center,available_RAM)
        dim_chunk_val = min(dim_chunk_val_input, dim_chunk_val_output)
    
        evaluator = SparseEvaluation(zonotope,
                                    chunk_size=dim_chunk_val, 
                                    function=function_epsilon, 
                                    mask_coef=mask_epsilon, 
                                    device= device)
        
        zonotope = evaluator.evaluate_all_chunks(num_workers=num_workers)
        len_zono = zonotope.size(0)
        mask_epsilon = torch.ones_like(mask_epsilon)
        
        if  add_symbol==True and not torch.equal(trash, torch.zeros_like(trash)):
            #_, new_sparse = ZonoSparseGeneration(trash,from_trash=True,start_index=len_zono).total_zono()
            new_sparse = ZonoSparseGeneration().zono_from_tensor(trash,start_index=len_zono).coalesce()
            if new_sparse is not None:
                print(f'new_sparse size {new_sparse.size()}')
                evaluator_new_noise = SparseEvaluation(new_sparse,
                                                    chunk_size = dim_chunk_val,
                                                    function=function_epsilon, 
                                                    mask_coef = mask_epsilon,
                                                    eval_start_index=len_zono,
                                                    device =device)
                new_sparse = evaluator_new_noise.evaluate_all_chunks(num_workers=num_workers)
                zonotope = torch.sparse_coo_tensor(zonotope.indices(), zonotope.values(), size = new_sparse.size()).coalesce()
                zonotope += new_sparse
                trash = static_process_trash_layer(trash,function_trash)
                trash = torch.zeros_like(trash)
        else : 
            trash = static_process_trash_layer(trash,function_trash)

        sum_abs = torch.sum(torch.abs(zonotope),dim=0).unsqueeze(0).to_dense()+torch.abs(trash)

        abstract_domain['zonotope'] = zonotope
        abstract_domain['center'] = center
        abstract_domain['sum'] = sum_abs
        abstract_domain['trash'] = trash
        abstract_domain['mask'] = torch.ones_like(center)
        abstract_domain['perfect_domain'] = True

              
        return abstract_domain