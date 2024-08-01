import torch
import torch.nn as nn
from typing import List, Union, Tuple
import numpy as np
import gc

class AbstractReLU(nn.Module):
    """
    Abstract ReLU layer for processing abstract domains.

    This class provides a method to evaluate an abstract domain with ReLU activation, modifying the domain's properties accordingly.

    Args:
        max_symbols (Union[int, bool], optional): Maximum number of symbols. Defaults to False.
    """
    def __init__(self, max_symbols: Union[int, bool] = False):
        super(AbstractReLU, self).__init__()

    @staticmethod
    def evaluate(abstract_domain, device: torch.device = torch.device("cpu")) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate the abstract domain with ReLU activation.

        Args:
            abstract_domain (dict): The abstract domain containing the zonotope, center, sum, trash, and mask tensors.
            device (torch.device, optional): The device to perform the computation on. Defaults to torch.device("cpu").

        Returns:
            dict: The updated abstract domain after ReLU activation.
        """
      
        center = abstract_domain['center']
        sum = abstract_domain['sum']
        trash = abstract_domain['trash']
        mask_epsilon = abstract_domain['mask']

        x_center = center.to(device)
        sum = sum.to(device)
        trash_layer = trash.to(device)
        x_min = x_center - sum - torch.abs(trash_layer)
        x_max = x_center + sum + torch.abs(trash_layer)

        sgn_min = torch.sign(x_min)
        sgn_max = torch.sign(x_max)
        sgn = sgn_min + sgn_max

        coef_approx_linear = x_max / (torch.abs(x_max) + torch.abs(x_min))
        coef_approx_linear = torch.where(torch.isnan(coef_approx_linear), torch.zeros_like(coef_approx_linear), coef_approx_linear)

        bias_approx_linear = x_max * (1 - coef_approx_linear) / 2
        noise_approx_linear = torch.abs(bias_approx_linear)

        mask_p = (sgn == 0)
        mask_1 = (sgn == 2) + (sgn == 1)
        mask_0 = (sgn == -2) + (sgn == -1)

        new_center = torch.ones_like(x_center)
        new_center[mask_1] = x_center[mask_1]
        new_center[mask_p] = coef_approx_linear[mask_p] * x_center[mask_p] + bias_approx_linear[mask_p]
        new_center[mask_0] = 0

        mask_epsilon = torch.zeros_like(x_center)
        mask_epsilon[mask_p] = coef_approx_linear[mask_p]
        mask_epsilon[mask_1] = 1

        trash_layer[mask_p] = noise_approx_linear[mask_p] + torch.abs(coef_approx_linear[mask_p]) * trash_layer[mask_p]
        trash_layer[mask_0] = 0

       
        abstract_domain['center'] = new_center
        abstract_domain['trash'] = trash_layer.to('cpu')
        abstract_domain['mask'] = mask_epsilon
        abstract_domain['perfect_domain'] = False

        return abstract_domain
