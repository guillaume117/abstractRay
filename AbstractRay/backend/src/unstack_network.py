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
        last_layer (int,optional): The deepht of layer choosen for analysis (e.g) if arg = 3, only the 3 first layers will be analysed
    """

    def __init__(self, model, input_dim, threshold=1e-5,last_layer = None):
        """
        Initialize the UnStackNetwork class.

        Args:
            model (nn.Module): The PyTorch model to unstack.
            input_dim (tuple): The dimensions of the input tensor.
            threshold (float, optional): Threshold for comparing outputs. Defaults to 1e-5.
            last_layer (int,optional): The deepht of layer choosen for analysis (e.g) if arg = 3, only the 3 first layers will be analysed
    
        """
        self.model = model
        self.input_dim = input_dim
        self.output = {}
        self.threshold = threshold
        self.last_layer = last_layer
        if last_layer== None : 
            self.last_layer=np.inf
        else :
            self.last_layer= last_layer
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
                        self.last_layer -=1
                        if self.last_layer ==0:
                            x = self.add_identity_layer(x)
                            return 
                else:
                    x = self.process_layer(name, module, x)
                self.last_layer -=1
                if self.last_layer ==0:
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
                'output_dim': self.compute_output_dim(layer, x)
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
                'output_dim': self.compute_output_dim(layer, x)
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
                'output_dim': self.compute_output_dim(layer, x)
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
                'output_dim': self.compute_output_dim(identity_layer, x)
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
