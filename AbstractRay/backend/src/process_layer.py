import torch
import torch.nn as nn
import sys
from AbstractRay.backend.src.process_abstract_maxpool2d_2 import process_max_pool2D
from AbstractRay.backend.src.process_linear_layer import static_process_linear_layer, static_process_linear_layer_parrallel
from AbstractRay.backend.src.abstract_relu import AbstractReLU

def process_layer(abstract_domain, name, details, num_worker, available_ram, device, add_symbol,parallel=False,evaluator_rel =None):
    """
    Process a layer within the abstract domain.

    Args:
        abstract_domain (dict): The abstract domain to process.
        name (str): The name of the layer.
        details (dict): The details of the layer, including the original, epsilon, and noise functions.
        num_worker (int): Number of workers for processing.
        available_ram (int): Available RAM in GB.
        device (torch.device): The device to run the processing on.
        add_symbol (bool): Flag to indicate whether to add a symbol.

    Returns:
        dict: The updated abstract domain after processing the layer.
    """
    linear_layer = details.get('original', None)
    activation_layer = details.get('activation', None)
    
    if linear_layer:
        function_center = details['original']
        function_epsilon = details[f'epsilon_{name}']
        function_trash = details[f'noise_{name}']
        if not parallel:
            abstract_domain = static_process_linear_layer(
                abstract_domain,
                function_center=function_center,
                function_epsilon=function_epsilon,
                function_trash=function_trash,
                num_workers=num_worker,
                available_RAM=available_ram,
                device=device,
                add_symbol=add_symbol
            )
            return abstract_domain
        if parallel:
            abstract_domain = static_process_linear_layer_parrallel(evaluator_rel=evaluator_rel,
                                                                    abstract_domain=abstract_domain,
                                                                    function_center=function_center,
                                                                    function_epsilon=function_epsilon,
                                                                    function_trash=function_trash,
                                                                    device=device, 
                                                                    available_RAM=available_ram,
                                                                    num_workers=num_worker,
                                                                    add_symbol=add_symbol)
            
            return abstract_domain
    if activation_layer:
        if activation_layer == 'MaxPool2d':
            abstract_domain = process_max_pool2D(
                abstract_domain=abstract_domain,
                evaluator_rel=evaluator_rel,
                maxpool_layer=details['activation_function'],
                num_workers=num_worker,
                available_ram=available_ram,
                device=device,
                add_symbol=add_symbol)
           
            return abstract_domain
        else:
            class_name = f"Abstract{activation_layer}"
            AbstractClass = globals().get(class_name)
            if AbstractClass:
                abstract_instance = AbstractClass
                abstract_domain = abstract_instance.evaluate(abstract_domain)
            
                return abstract_domain
