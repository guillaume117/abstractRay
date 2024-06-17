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
        x = torch.randn(1, *self.input_dim)
        for name, module in self.model.named_children():
            if isinstance(module, nn.Sequential):
                for layer_name, layer in module.named_children():
                    x = self.process_layer(f"{name}_{layer_name}", layer, x)
            else:
                x = self.process_layer(name, module, x)

    def process_layer(self, name, layer, x):
        if isinstance(layer, nn.Linear):
            # Flatten the tensor before passing to fully connected layers
            x = x.view(x.size(0), -1)
            self.process_linear_layer(name, layer, x)
        elif isinstance(layer, nn.Conv2d):
            self.process_conv_layer(name, layer, x)
        elif isinstance(layer, (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.AdaptiveAvgPool2d, nn.MaxPool2d)):
            self.process_activation_layer(name, layer)
        x = layer(x) if layer is not None else x  # Handle input layer without layer
        return x

    def process_linear_layer(self, name, layer, x):
        self.output[name] = {
            'type': type(layer),
            'original': copy.deepcopy(layer),
            'epsilon_{}'.format(name): self.copy_with_zero_bias(layer),
            'noise_{}'.format(name): self.copy_with_abs_weights(layer),
            'output_dim': self.compute_output_dim(layer, x)
        }

    def process_conv_layer(self, name, layer, x):
        self.output[name] = {
            'type': type(layer),
            'original': copy.deepcopy(layer),
            'epsilon_{}'.format(name): self.copy_with_zero_bias(layer),
            'noise_{}'.format(name): self.copy_with_abs_weights(layer),
            'output_dim': self.compute_output_dim(layer, x)
        }

    def process_activation_layer(self, name, layer):
        self.output[name] = {
            'activation': layer.__class__.__name__
        }

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
            out = layer(x) if layer is not None else x  # Handle input layer without layer
        return out.shape

# Charger le modèle VGG19 pré-entraîné
model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)

# Spécifier les dimensions d'entrée (par exemple, une image RGB de 224x224)
input_dim = (3, 224, 224)

# Utiliser la classe UnStackNetwork
unstacked = UnStackNetwork(model, input_dim)

# Afficher les résultats
print(unstacked.output)
