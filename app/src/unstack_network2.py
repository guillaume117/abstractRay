import torch
import torch.nn as nn
import torchvision.models as models
import copy

class UnStackNetwork:
    def __init__(self, model, input_dim, threshold=1e-5):
        self.model = model
        self.input_dim = input_dim
        self.output = {}
        self.threshold = threshold
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

            # Ajouter une couche Linear identité à la fin du réseau
            x = self.add_identity_layer(x)

    def process_layer(self, name, layer, x):
        with torch.no_grad():
            original_output = x.clone()
            if isinstance(layer, nn.Linear):
                x = nn.Flatten()(x)
                original_output = nn.Flatten()(original_output)
                x = self.process_linear_layer(name, nn.Sequential(nn.Flatten(), layer), x)
            elif isinstance(layer, nn.Conv2d):
                x = self.process_conv_layer(name, layer, x)
            elif isinstance(layer, nn.AdaptiveAvgPool2d):
                x = self.process_avgpool_layer(name, layer, x)
                print(x.shape)
            elif isinstance(layer, (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.MaxPool2d)):
                x = self.process_activation_layer(name, layer, x)
            elif isinstance(layer, nn.Flatten):
                x = self.process_flatten(name, layer, x)
            elif isinstance(layer, nn.Sequential):
                x = self.process_sequential(name, layer, x)
            else:
                if hasattr(layer, 'forward'):
                    x = layer(x)
                else:
                    print(f"Layer {name} not processed")

            if isinstance(x, tuple):
                x = x[0]

            self.compare_outputs(layer(original_output), x, name)

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
                'activation_function': layer
            }
            return layer(x)

    def process_sequential(self, name, layer, x):
        with torch.no_grad():
            for sub_name, sub_module in layer.named_children():
                x = self.process_layer(f"{name}.{sub_name}", sub_module, x)
            return x

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

    def compare_outputs(self, original_output, new_output, layer_name):
        original_output_flat = original_output.view(-1)
        new_output_flat = new_output.view(-1)
        min_length = min(len(original_output_flat), len(new_output_flat))
        difference = torch.abs(original_output_flat[:min_length] - new_output_flat[:min_length]).max().item()
        if difference > self.threshold:
            raise ValueError(f"Difference between original and new output is too high after layer {layer_name}: {difference}")
        print(f"Output comparison after layer {layer_name} passed")

# Exemple d'utilisation
model = models.resnet18(pretrained=False)
model.eval()
input_dim = (3, 224, 224)  # Utiliser la taille d'entrée correcte pour ResNet
unstacked_network = UnStackNetwork(model, input_dim)
print(unstacked_network.output)
