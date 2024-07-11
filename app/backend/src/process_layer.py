import torch
import torch.nn as nn
import sys
sys.path.append('app/src')
sys.path.append('./src')
from util import sparse_tensor_stats, get_largest_tensor_size,sparse_dense_broadcast_mult, resize_sparse_coo_tensor

from process_abstract_maxpool2d_layer import process_max_pool2D
from process_linear_layer import static_process_linear_layer
from sparse_evaluation_4 import SparseEvaluation  
from zono_sparse_gen import ZonoSparseGeneration
from abstract_relu import AbstractReLU
from sparse_addition_2 import SparseAddition
from process_abstract_maxpool2d_layer import process_max_pool2D


def process_layer(abstract_domain, name, details, num_worker, available_ram, device, add_symbol):
    """
    """
    linear_layer = details.get('original',None)
    activation_layer = details.get('activation',None)
    if linear_layer:

        function_center = details['original']
        function_epsilon = details[f'epsilon_{name}']
        function_trash = details[f'noise_{name}']
        abstract_domain = static_process_linear_layer(abstract_domain,
                                                        function_center=function_center,
                                                        function_epsilon=function_epsilon,
                                                        function_trash= function_trash,
                                                        num_workers=num_worker,
                                                        available_RAM=available_ram,
                                                        device=device,
                                                        add_symbol=add_symbol)
        return abstract_domain
    
    if activation_layer:
        if activation_layer == 'MaxPool2d':
            abstract_domain = process_max_pool2D(abstract_domain=abstract_domain,maxpool_layer=details['activation_function'],num_workers=num_worker,available_ram=available_ram,device=device,add_symbol=add_symbol)
            return abstract_domain
        else :
            class_name = f"Abstract{activation_layer}"
            print(class_name)
            AbstractClass = globals().get(class_name)
            if AbstractClass:
                abstract_instance = AbstractClass
                abstract_domain = abstract_instance.evaluate(abstract_domain)
                return abstract_domain
