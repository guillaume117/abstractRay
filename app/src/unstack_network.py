import torch
import torch.nn as nn
import torchvision.models as models
import copy
import onnx



class UnStackNetwork:
    with torch.no_grad():
        def __init__(self, model, input_dim):
            self.model = model
            self.input_dim = input_dim
            self.output = {}
            self.unstack_network()

        def unstack_network(self):
            x = torch.randn(1, *self.input_dim)
            for name, module in self.model.named_children():
                print(x.size())
                x = self.process_layer(name, module, x)


        def process_layer(self, name, layer, x):
            print(layer)
            if isinstance(layer, nn.Linear):
                #x = self.process_flatten(name, nn.Flatten(), x)
                x = nn.Flatten()(x)
                x = self.process_linear_layer(name, nn.Sequential(nn.Flatten(),layer), x)
            elif isinstance(layer, nn.Conv2d):
                x = self.process_conv_layer(name, layer, x)
            elif isinstance(layer, (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.AdaptiveAvgPool2d, nn.MaxPool2d)):
                x = self.process_activation_layer(name, layer, x)
            elif isinstance(layer, nn.Flatten):
                x = self.process_flatten(name, layer, x)
            
            
            else:
                if layer is not None and not 'Module()' and not 'OnnxDropoutDynamic()':
                    print(layer)
                    x = layer(x)   # Handle input layer without layer
                

            # Handle tuple outputs
            if isinstance(x, tuple):
                x = x[0]

            return x
        def process_linear_layer(self, name, layer, x):
            self.output[name] = {
                'type': type(layer),
                'original': copy.deepcopy(layer),
                'epsilon_{}'.format(name): self.copy_with_zero_bias(layer),
                'noise_{}'.format(name): self.copy_with_abs_weights(layer),
                'output_dim': self.compute_output_dim(layer, x)
            }
            return layer(x)

        def process_flatten(self, name, layer, x):
            flattened_x = layer(x)  # Apply the flatten layer to get the output dimensions
            self.output[f'{name}_flatten'] = {
                'type': type(layer),
                'original': layer,
                'epsilon_{}'.format(f'{name}_flatten'): layer,
                'noise_{}'.format(f'{name}_flatten'): layer,
                'output_dim': self.compute_output_dim(layer, flattened_x)
            }
            return flattened_x

        def process_conv_layer(self, name, layer, x):
            self.output[name] = {
                'type': type(layer),
                'original': copy.deepcopy(layer),
                'epsilon_{}'.format(name): self.copy_with_zero_bias(layer),
                'noise_{}'.format(name): self.copy_with_abs_weights(layer),
                'output_dim': self.compute_output_dim(layer, x)
            }
            return layer(x)

        def process_activation_layer(self, name, layer, x):
            self.output[name] = {
                'activation': layer.__class__.__name__
            }
            print(self.output[name])
            print(type(x))
            return layer(x)

        def copy_with_zero_bias(self, layer):
            new_layer = copy.deepcopy(layer)
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                with torch.no_grad():
                    if new_layer.bias is not None:
                        new_layer.bias.zero_()
            return new_layer

        def copy_with_abs_weights(self, layer):
            new_layer = copy.deepcopy(layer)
            with torch.no_grad():
                if isinstance(layer, (nn.Linear, nn.Conv2d)):
                    if new_layer.bias is not None:
                        new_layer.bias.zero_()
                    new_layer.weight.abs_()
            return new_layer

        def compute_output_dim(self, layer, x):
            with torch.no_grad():
                print(x.size())
                out = layer(x) if layer is not None else x  
            return out.shape
