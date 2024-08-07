
import torch
import torch.nn as nn
import copy
import numpy as np
import torch
import torch.nn as nn
from typing import List, Union, Tuple
import numpy as np
import time 
import torch
import torch.nn as nn
import copy
import numpy as np

class UnStackNetwork:
    """
    Unstack a PyTorch model and store information about each layer.

    Attributes:
        model (nn.Module): The PyTorch model to unstack.
        input_dim (tuple): The dimensions of the input tensor.
        output (dict): Dictionary to store information about each layer.
        threshold (float): Threshold for comparing outputs.
        last_layer (int, optional): The depth of layer chosen for analysis (e.g., if arg = 3, only the first 3 layers will be analyzed).
    """

    def __init__(self, model, input_dim, threshold=1e-5, last_layer=None):
        """
        Initialize the UnStackNetwork class.

        Args:
            model (nn.Module): The PyTorch model to unstack.
            input_dim (tuple): The dimensions of the input tensor.
            threshold (float, optional): Threshold for comparing outputs. Defaults to 1e-5.
            last_layer (int, optional): The depth of layer chosen for analysis (e.g., if arg = 3, only the first 3 layers will be analyzed).
        """
        self.model = model
        self.input_dim = input_dim
        self.output = {}
        self.threshold = threshold
        self.last_layer = last_layer
        if last_layer is None:
            self.last_layer = np.inf
        else:
            self.last_layer = last_layer
        self.unstack_network()

    def unstack_network(self):
        """
        Unstack the network by passing a random tensor through the model and processing each layer.
        """
        with torch.no_grad():
            x = torch.randn(1, *self.input_dim)

            for name, module in self.model.named_children():
                if isinstance(module, nn.Sequential):
                    for sub_name, sub_module in module.named_children():
                        x = self.process_layer(f"{name}.{sub_name}", sub_module, x)
                        self.last_layer -= 1
                        if self.last_layer == 0:
                            x = self.add_identity_layer(x)
                            return
                else:
                    x = self.process_layer(name, module, x)
                self.last_layer -= 1
                if self.last_layer == 0:
                    x = self.add_identity_layer(x)
                    return

            x = self.add_identity_layer(x)
            return

    def process_layer(self, name, layer, x):
        """
        Process a single layer of the network.

        Args:
            name (str): The name of the layer.
            layer (nn.Module): The layer to process.
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after processing the layer.
        """
        with torch.no_grad():
            original_output = x.clone()
            if isinstance(layer, nn.Linear):
                x = nn.Flatten()(x)
                original_output = nn.Flatten()(original_output)
                x = self.process_linear_layer(name, nn.Sequential(nn.Flatten(), layer), x)
                self.compare_outputs(layer(original_output), x, name)
            elif isinstance(layer, nn.Conv2d):
                x = self.process_conv_layer(name, layer, x)
            elif isinstance(layer, nn.AdaptiveAvgPool2d):
                x = self.process_avgpool_layer(name, layer, x)
                self.compare_outputs(layer(original_output), x, name)
            elif isinstance(layer, (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.MaxPool2d)):
                x = self.process_activation_layer(name, layer, x)
                self.compare_outputs(layer(original_output), x, name)
            elif isinstance(layer, nn.Flatten):
                x = self.process_flatten(name, layer, x)
                self.compare_outputs(layer(original_output), x, name)
            elif isinstance(layer, nn.Sequential):
                x = self.process_sequential(name, layer, x)
                self.compare_outputs(layer(original_output), x, name)
            else:
                if hasattr(layer, 'forward'):
                    if layer is not None and not 'Module()' and not 'OnnxDropoutDynamic()':
                        x = layer(x)
                        print(f'layer {name} passed')
                        self.compare_outputs(layer(original_output), x, name)
                    else:
                        pass
                else:
                    print(f"Layer {name} not processed")
                    self.compare_outputs(original_output, x, name)
            if isinstance(x, tuple):
                x = x[0]
            return x

    def process_linear_layer(self, name, layer, x):
        """
        Process a linear layer of the network.

        Args:
            name (str): The name of the layer.
            layer (nn.Module): The linear layer to process.
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after processing the layer.
        """
        with torch.no_grad():
            self.output[name] = {
                'type': type(layer),
                'original': copy.deepcopy(layer),
                'epsilon_{}'.format(name): self.copy_with_zero_bias(layer),
                'noise_{}'.format(name): self.copy_with_abs_weights(layer),
                'output_dim': self.compute_output_dim(layer, x),
                'algebric_representation': self.determine_matrix_linear(self.copy_with_zero_bias(layer))
            }
            return layer(x)

    def process_flatten(self, name, layer, x):
        """
        Process a flatten layer of the network.

        Args:
            name (str): The name of the layer.
            layer (nn.Module): The flatten layer to process.
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after processing the layer.
        """
        with torch.no_grad():
            flattened_x = layer(x)
            self.output[f'{name}_flatten'] = {
                'type': type(layer),
                'original': layer,
                'epsilon_{}'.format(f'{name}_flatten'): layer,
                'noise_{}'.format(f'{name}_flatten'): layer,
                'output_dim': self.compute_output_dim(layer, flattened_x)
            }
            return flattened_x

    def process_conv_layer(self, name, layer, x):
        """
        Process a convolutional layer of the network.

        Args:
            name (str): The name of the layer.
            layer (nn.Module): The convolutional layer to process.
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after processing the layer.
        """
        with torch.no_grad():
            self.output[name] = {
                'type': type(layer),
                'original': copy.deepcopy(layer),
                'epsilon_{}'.format(name): self.copy_with_zero_bias(layer),
                'noise_{}'.format(name): self.copy_with_abs_weights(layer),
                'output_dim': self.compute_output_dim(layer, x),
                'algebric_representation': self.determine_matrix(lambda y: self.copy_with_zero_bias(layer)(y), x, layer(x))
            }
            return layer(x)

    def process_avgpool_layer(self, name, layer, x):
        """
        Process an adaptive average pooling layer of the network.

        Args:
            name (str): The name of the layer.
            layer (nn.Module): The adaptive average pooling layer to process.
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after processing the layer.
        """
        with torch.no_grad():
            self.output[name] = {
                'type': type(layer),
                'original': copy.deepcopy(layer),
                'epsilon_{}'.format(name): copy.deepcopy(layer),
                'noise_{}'.format(name): copy.deepcopy(layer),
                'output_dim': self.compute_output_dim(layer, x),
                'A': self.determine_matrix(lambda y: layer(y), x, layer(x))
            }
            return layer(x)

    def process_activation_layer(self, name, layer, x):
        """
        Process an activation layer of the network.

        Args:
            name (str): The name of the layer.
            layer (nn.Module): The activation layer to process.
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after processing the layer.
        """
        with torch.no_grad():
            self.output[name] = {
                'activation': layer.__class__.__name__,
                'activation_function': layer
            }
            return layer(x)

    def process_sequential(self, name, layer, x):
        """
        Process a sequential container of the network.

        Args:
            name (str): The name of the layer.
            layer (nn.Sequential): The sequential container to process.
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after processing the layer.
        """
        with torch.no_grad():
            for sub_name, sub_module in layer.named_children():
                x = self.process_layer(f"{name}.{sub_name}", sub_module, x)
            return x

    def add_identity_layer(self, x):
        """
        Add an identity layer to the network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after adding the identity layer.
        """
        with torch.no_grad():
            identity_layer = nn.Identity()
            self.output['identity_layer'] = {
                'type': type(identity_layer),
                'original': identity_layer,
                'epsilon_identity_layer': identity_layer,
                'noise_identity_layer': identity_layer,
                'output_dim': self.compute_output_dim(identity_layer, x),
                'algebric_representation': self.determine_matrix(lambda y: y, x, x)
            }
            return identity_layer(x)

    def copy_with_zero_bias(self, layer):
        """
        Create a copy of a layer with zeroed biases.

        Args:
            layer (nn.Module): The layer to copy.

        Returns:
            nn.Module: The copied layer with zeroed biases.
        """
        with torch.no_grad():
            new_layer = copy.deepcopy(layer)
            if isinstance(new_layer, (nn.Linear, nn.Conv2d)):
                if new_layer.bias is not None:
                    new_layer.bias.zero_()
            elif isinstance(new_layer, nn.Sequential):
                for sublayer in new_layer:
                    if isinstance(sublayer, (nn.Linear, nn.Conv2d)):
                        if sublayer.bias is not None:
                            sublayer.bias.zero_()
            return new_layer

    def copy_with_abs_weights(self, layer):
        """
        Create a copy of a layer with absolute weights.

        Args:
            layer (nn.Module): The layer to copy.

        Returns:
            nn.Module: The copied layer with absolute weights.
        """
        with torch.no_grad():
            new_layer = copy.deepcopy(layer)
            if isinstance(new_layer, (nn.Linear, nn.Conv2d)):
                if new_layer.bias is not None:
                    new_layer.bias.zero_()
                new_layer.weight.abs_()
            elif isinstance(new_layer, nn.Sequential):
                for sublayer in new_layer:
                    if isinstance(sublayer, (nn.Linear, nn.Conv2d)):
                        if sublayer.bias is not None:
                            sublayer.bias.zero_()
                        sublayer.weight.abs_()
            return new_layer

    def compute_output_dim(self, layer, x):
        """
        Compute the output dimensions of a layer.

        Args:
            layer (nn.Module): The layer for which to compute output dimensions.
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Size: The output dimensions of the layer.
        """
        with torch.no_grad():
            out = layer(x) if layer is not None else x
            return out.shape

    def determine_matrix(self, function_epsilon, m, p):
        """
        Determine the representation matrix of a linear function f: R^m -> R^p.

        Args:
            function_epsilon (function): The function representing the layer.
            m (torch.Tensor): The input tensor.
            p (torch.Tensor): The output tensor.

        Returns:
            torch.Tensor: The representation matrix of the function.
        """
        m_in = m.numel()
        p_out = p.numel()
        model = lambda x: function_epsilon(x)
        
        # Indices and values for the sparse matrix
        indices = []
        values = []

        for i in range(m_in):
            # Create base vector dynamically
            base_vector = torch.zeros(m_in)
            base_vector[i] = 1.0
            
            result = model(base_vector.unsqueeze(0).view_as(m)).flatten()
            non_zero_indices = torch.nonzero(result, as_tuple=True)[0]
            non_zero_values = result[non_zero_indices]
            
            if non_zero_values.numel() > 0:
                indices.append(torch.stack([non_zero_indices, torch.full_like(non_zero_indices, i)], dim=0))
                values.append(non_zero_values)

        if indices:
            indices = torch.cat(indices, dim=1)
            values = torch.cat(values)
        else:
            indices = torch.empty((2, 0), dtype=torch.long)
            values = torch.empty(0, dtype=m.dtype)
        
        algebric_representation_sparse = torch.sparse_coo_tensor(indices, values, (p_out, m_in)).coalesce()
        
        return algebric_representation_sparse
    

    def determine_matrix_linear(self, function_epsilon):
        """
        Determine the representation matrix of a linear function f: R^m -> R^p.
        In this case, a copy from nn.Linear.weight

        Args:
            function_epsilon (function): The function representing the layer.
            m (torch.Tensor): The input tensor.
            p (torch.Tensor): The output tensor.

        Returns:
            torch.Tensor: The algebricrepresentation matrix of the function.
        """
        if isinstance(function_epsilon, nn.Linear):
       
            return function_epsilon.weight.data.to_sparse_coo()
        elif isinstance(function_epsilon, nn.Sequential):
            for sublayer in function_epsilon:
                if isinstance(sublayer, nn.Linear):
                    return sublayer.weight.data.to_sparse_coo()    

    def compare_outputs(self, original_output, new_output, layer_name):
        """
        Compare the outputs of the original and modified layers.

        Args:
            original_output (torch.Tensor): The output of the original layer.
            new_output (torch.Tensor): The output of the modified layer.
            layer_name (str): The name of the layer being compared.

        Raises:
            ValueError: If the difference between the original and new output exceeds the threshold.
        """
        original_output_flat = original_output.view(-1)
        new_output_flat = new_output.view(-1)
        min_length = min(len(original_output_flat), len(new_output_flat))
        difference = torch.abs(original_output_flat[:min_length] - new_output_flat[:min_length]).max().item()
        if difference > self.threshold:
            raise ValueError(f"Difference between original and new output is too high after layer {layer_name}: {difference}")
        print(f"Output comparison after layer {layer_name} passed")



"""
This is the D
"""
class DummyCNN(nn.Module):
    
    def __init__(self):
        super(DummyCNN,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3)
        self.relu2 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=73728,out_features=10)
        self.relu3 = nn.ReLU()
    def forward(self,x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.relu3(x)

        return x 


list_algebric_evaluation =[]


def static_process_linear_layer(abstract_domain, function_center,algebric_representation,list_algebric_evaluation):
    """
    Process a linear layer within the abstract domain.

    Args:
        abstract_domain (dict): The abstract domain to process.
        function_center (nn.Module): The function to process the center tensor

        add_symbol (bool, optional): Flag to indicate whether to add a symbol. Defaults to True.

    Returns:
        dict: The updated abstract domain after processing the linear layer.
    """

    center = abstract_domain['center']
    sum_abs = abstract_domain['sum']
    trash = abstract_domain['trash']
    mask= abstract_domain['mask']

    with torch.no_grad():

        center = function_center(center)
        sum_abs = torch.zeros_like(center).flatten()
        if algebric_representation is not None: 
            A_eval = mask.flatten()*algebric_representation
            list_algebric_evaluation = [A_eval@mat for mat in list_algebric_evaluation]
            list_algebric_evaluation.append(algebric_representation)
            list_algebric_evaluation_absolute = [torch.abs(mat)for mat in list_algebric_evaluation] 
            for trash, algebric_representation_absolute in zip(abstract_domain['zonotope'],list_algebric_evaluation_absolute):
                sum_abs += torch.sparse.mm(algebric_representation_absolute, torch.abs(trash).unsqueeze(1)).squeeze()
            mask = torch.ones_like(mask)
        sum_abs= sum_abs.view_as(center)
        
       
        abstract_domain['center'] = center
        abstract_domain['sum'] = sum_abs
        abstract_domain['trash'] = torch.zeros_like(center)
        abstract_domain['mask'] = mask
        abstract_domain['perfect_domain'] = True
  

        return abstract_domain,list_algebric_evaluation
    

def process_layer(abstract_domain, name, details,list_algebric_evaluation):
    """
    Process a layer within the abstract domain.

    Args:
        abstract_domain (dict): The abstract domain to process.
        name (str): The name of the layer.
        details (dict): The details of the layer, including the original, epsilon, and noise functions.

    Returns:
        dict: The updated abstract domain after processing the layer.
    """
    linear_layer = details.get('original', None)
    activation_layer = details.get('activation', None)
    algebric_representation = details.get('algebric_representation',None)
    
    if linear_layer:
        function_center = details['original']
     
   
        abstract_domain,list_algebric_evaluation = static_process_linear_layer(
            abstract_domain,
            function_center=function_center,
            algebric_representation=algebric_representation,
            list_algebric_evaluation=list_algebric_evaluation
            )
     
        return abstract_domain,list_algebric_evaluation

    if activation_layer: 
        class_name = f"Abstract{activation_layer}"
        AbstractClass = globals().get(class_name)
        if AbstractClass:
            abstract_instance = AbstractClass
            abstract_domain = abstract_instance.evaluate(abstract_domain)
        
            return abstract_domain,list_algebric_evaluation
        


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
        abstract_domain['zonotope'].append(trash_layer.flatten())
        abstract_domain['mask'] = mask_epsilon


        return abstract_domain






"""Here we initialize network and tensor for evaluations"""

model = DummyCNN()
unstack = UnStackNetwork(model=model,input_dim=(3,28,28))
x=torch.randn(1,3,28,28)
noise_level = 0.1
""" 
Here we test the method with the latent algebric reprsentation of the affine forms 
"""
print("Here we test the method with the latent algebric reprsentation of the affine forms")

for i in range(1):
    list_algebric_evaluation =[]
    abstract_domain = {
        'zonotope': [noise_level*torch.ones_like(x).flatten()],
        'center': x,
        'sum': torch.zeros_like(x),
        'trash': torch.zeros_like(x),
        'mask': torch.ones_like(x),
        'perfect_domain': True
    }

    time_start = time.time()
    for name, details in unstack.output.items():
        abstract_domain,list_algebric_evaluation =process_layer(abstract_domain, name, details,list_algebric_evaluation=list_algebric_evaluation) 
    time_end = time.time()


argmax = torch.topk(model(x).squeeze(0), 10).indices

response = {
    "argmax": argmax.tolist(),
    "true": model(x).squeeze(0)[argmax].tolist(),
    "center": abstract_domain['center'].squeeze(0)[argmax].tolist(),
    "min": (abstract_domain['center'].squeeze(0)[argmax] - abstract_domain['sum'].squeeze(0)[argmax]-abstract_domain['trash'].squeeze(0)[argmax]).tolist(),
    "max": (abstract_domain['center'].squeeze(0)[argmax] + abstract_domain['sum'].squeeze(0)[argmax]+abstract_domain['trash'].squeeze(0)[argmax]).tolist(),
    "diff_center_true": torch.max(model(x) - abstract_domain['center']).item()
}

print("Evaluation Results Algebric representation method:")
for key, value in response.items():
    print(f"{key}: {value}")
print(f"Execution time = {time_end-time_start}")
print("#"*100)


""" 
Here we test the method with the classical reprsentation of the affine forms 
"""
print("Here we test the method with the classical representation of the affine forms")

#Ici on compare les résultas avec la méthode standard d'évaluation
from AbstractRay.backend.src.zono_sparse_gen import ZonoSparseGeneration
from AbstractRay.backend.src.evaluator import ModelEvaluator
from AbstractRay.backend.src.unstack_network import UnStackNetwork
from AbstractRay.backend.src.util import ensure_ray_initialized
unstack_network = UnStackNetwork(model, (3,28,28))

zonotope_espilon_sparse_tensor = ZonoSparseGeneration().zono_from_noise_level_and_tensor(noise_level=noise_level, tensor=x)
abstract_domain = {
                'zonotope': zonotope_espilon_sparse_tensor,
                'center': x,
                'sum': torch.zeros_like(x),
                'trash': torch.zeros_like(x),
                'mask': torch.ones_like(x),
                'perfect_domain': True
            }

model_evaluator = ModelEvaluator(
                unstack_network.output,
                abstract_domain,
                num_workers=0,
                available_RAM=1200,
                device=torch.device('cpu'),
                add_symbol=True,
                json_file_prefix='test',
                noise_level=noise_level,
                renew_abstract_domain=False,
                parrallel_rel=False,
                verbose=False
            )
time_start = time.time()
abstract_domain = model_evaluator.evaluate_model()
time_end = time.time()
argmax = torch.topk(model(x).squeeze(0), 10).indices

response = {
    "argmax": argmax.tolist(),
    "true": model(x).squeeze(0)[argmax].tolist(),
    "center": abstract_domain['center'].squeeze(0)[argmax].tolist(),
    "min": (abstract_domain['center'].squeeze(0)[argmax] - abstract_domain['sum'].squeeze(0)[argmax]-abstract_domain['trash'].squeeze(0)[argmax]).tolist(),
    "max": (abstract_domain['center'].squeeze(0)[argmax] + abstract_domain['sum'].squeeze(0)[argmax]+abstract_domain['trash'].squeeze(0)[argmax]).tolist(),
    "diff_center_true": torch.max(model(x) - abstract_domain['center']).item()
}

print("Evaluation Results classical method:")
for key, value in response.items():
    print(f"{key}: {value}")
print(f"Execution time = {time_end-time_start}")
print("#"*100)
print("If the results are the same, that means that you'll have to work a lot for recoding Saimple, good luke guys:)")



