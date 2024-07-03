import torch
import torch.nn as nn
import torchvision.models as models
import copy

class UnStackNetwork:
    def __init__(self, model, input_dim):
        self.model = model
        self.input_dim = input_dim
        self.output = {}
        self.unstack_network()

    def unstack_network(self):
        with torch.no_grad():
            x = torch.randn(1, *self.input_dim)

            for name, module in self.model.named_children():
                if isinstance(module, nn.Sequential):
                    for sub_name, sub_module in module.named_children():
                        x = self.process_layer(f"{name}.{sub_name}", sub_module, x)
                else:
                    x = self.process_layer(name, module, x)
            return x

        
        # Ajouter une couche Linear identité à la fin du réseau
        x = self.add_identity_layer(x)

    def process_layer(self, name, layer, x):
        with torch.no_grad():
   
            if isinstance(layer, nn.Linear):
                x = nn.Flatten()(x)
                x = self.process_linear_layer(name, nn.Sequential(nn.Flatten(), layer), x)
            elif isinstance(layer, nn.Conv2d):
                x = self.process_conv_layer(name, layer, x)

            elif isinstance(layer,nn.AdaptiveAvgPool2d):
                x  = self.process_avgpool_layer(name,layer,x)
                print(x.shape)
            elif isinstance(layer, (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.MaxPool2d)):
                x = self.process_activation_layer(name, layer, x)
            elif isinstance(layer, nn.Flatten):
                x = self.process_flatten(name, layer, x)
            else:
                if layer is not None and not 'Module()' and not 'OnnxDropoutDynamic()' and not 'OnnxShape()' and not 'OnnxGather' and not 'OnnxConstant' and not 'OnnxConcat':
            
                    x = layer(x)   # Handle input layer without layer

            # Handle tuple outputs
            if isinstance(x, tuple):
                x = x[0]

            return x

    def process_linear_layer(self, name, layer, x):
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
        with torch.no_grad():
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
        with torch.no_grad():
            self.output[name] = {
                'activation': layer.__class__.__name__,
                'activation_function':layer
            }
     
      
            return layer(x)

    def add_identity_layer(self, x):
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
        with torch.no_grad():
            new_layer = copy.deepcopy(layer)
            with torch.no_grad():
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
        with torch.no_grad():
            new_layer = copy.deepcopy(layer)
            with torch.no_grad():
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
        with torch.no_grad():
         
            out = layer(x) if layer is not None else x  
            return out.shape


