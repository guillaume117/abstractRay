import torch
import torch.nn as nn
import torchvision.models as models
import copy
class Concat(nn.Module):
    def __init__(self, dim=1):
        super(Concat, self).__init__()
        self.dim = dim

    def forward(self, *inputs):
        return torch.cat(inputs, dim=self.dim)
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
            self.hooks = []

            # Enregistrement des hooks pour capturer les sorties intermédiaires
            for name, module in self.model.named_modules():
                self.hooks.append(module.register_forward_hook(self.save_output_hook(name)))

            # Traiter chaque couche
            for name, module in self.model.named_children():
                x = self.process_layer(name, module, x)

            # Supprimer les hooks après l'exécution
            for hook in self.hooks:
                hook.remove()

            # Ajouter une couche Linear identité à la fin du réseau
            x = self.add_identity_layer(x)

    def save_output_hook(self, name):
        def hook(module, input, output):
            self.output[name] = output
            print(f"Hook: {name} - output shape: {output.shape}")
        return hook

    def process_layer(self, name, layer, x):
        with torch.no_grad():
            original_output = x.clone()
            print(layer)
            if isinstance(layer, nn.Linear):
                x = nn.Flatten()(x)
                original_output = nn.Flatten()(original_output)
                x = self.process_linear_layer(name, nn.Sequential(nn.Flatten(), layer), x)
            elif isinstance(layer, nn.Conv2d):
                x = self.process_conv_layer(name, layer, x)
            elif isinstance(layer,nn.ConvTranspose2d):
                x = self.process_upconv_layer(name,layer,x)
            elif isinstance(layer, nn.AdaptiveAvgPool2d):
                x = self.process_avgpool_layer(name, layer, x)
                print(x.shape)
            elif isinstance(layer, (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.MaxPool2d)):
                x = self.process_activation_layer(name, layer, x)
            elif isinstance(layer, nn.Flatten):
                x = self.process_flatten(name, layer, x)
            elif isinstance(layer, nn.Sequential):
                x = self.process_sequential(name, layer, x)
            elif isinstance(layer, Concat):
                inputs = [self.output[input_name] for input_name in layer.input_names]
                x = layer(*inputs)
                print(f'layer {name} passed')
            else:
                if hasattr(layer, 'forward'):
                    x = layer(x)
                    print(f'layer {name} passed')
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
        
    def process_upconv_layer(self, name, layer, x):
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

# Exemple d'utilisation avec ResNet
#resnet_model = models.resnet18(pretrained=False)
#resnet_model.eval()
#resnet_input_dim = (3, 224, 224)  # Utiliser la taille d'entrée correcte pour ResNet
#unstacked_resnet = UnStackNetwork(resnet_model, resnet_input_dim)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu

import torch
import torch.nn as nn
import torch.nn.functional as F

class Concat(nn.Module):
    def __init__(self, dim=1, input_names=None):
        super(Concat, self).__init__()
        self.dim = dim
        self.input_names = input_names if input_names is not None else []

    def forward(self, *inputs):
        return torch.cat(inputs, dim=self.dim)

class UNet(nn.Module):
    def __init__(self, n_class, num_dates, verbose=False):
        super().__init__()
        self.num_dates = num_dates
        self.n_classes = n_class

        self.e11 = nn.Conv2d(10, 64, kernel_size=3, padding=1)
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)

        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.concat1 = Concat(dim=1, input_names=['upconv1', 'e42'])
        self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.concat2 = Concat(dim=1, input_names=['upconv2', 'e32'])
        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.concat3 = Concat(dim=1, input_names=['upconv3', 'e22'])
        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.concat4 = Concat(dim=1, input_names=['upconv4', 'e12'])
        self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=0.05)

        self.outconv = nn.Conv2d(64, n_class * num_dates, kernel_size=1)
        self.verbose = verbose

    def forward(self, x):
        # Encoder
        xe11 = F.relu(self.e11(x))
        xe11 = self.dropout(xe11)
        xe12 = F.relu(self.e12(xe11))
        xe12 = self.dropout(xe12)
        xp1 = self.pool1(xe12)
        
        xe21 = F.relu(self.e21(xp1))
        xe21 = self.dropout(xe21)
        xe22 = F.relu(self.e22(xe21))
        xe22 = self.dropout(xe22)
        xp2 = self.pool2(xe22)
        
        xe31 = F.relu(self.e31(xp2))
        xe31 = self.dropout(xe31)
        xe32 = F.relu(self.e32(xe31))
        xe32 = self.dropout(xe32)
        xp3 = self.pool3(xe32)
        
        xe41 = F.relu(self.e41(xp3))
        xe41 = self.dropout(xe41)
        xe42 = F.relu(self.e42(xe41))
        xe42 = self.dropout(xe42)
        xp4 = self.pool4(xe42)
        
        xe51 = F.relu(self.e51(xp4))
        xe52 = F.relu(self.e52(xe51))

        # Decoder
        xu1 = self.upconv1(xe52)
        xu11 = self.concat1(xu1, xe42)
        xd11 = F.relu(self.d11(xu11))
        xd12 = F.relu(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        xu22 = self.concat2(xu2, xe32)
        xd21 = F.relu(self.d21(xu22))
        xd22 = F.relu(self.d22(xd21))

        xu3 = self.upconv3(xd22)
        xu33 = self.concat3(xu3, xe22)
        xd31 = F.relu(self.d31(xu33))
        xd32 = F.relu(self.d32(xd31))

        xu4 = self.upconv4(xd32)
        xu44 = self.concat4(xu4, xe12)
        xd41 = F.relu(self.d41(xu44))
        xd42 = F.relu(self.d42(xd41))

        # Output layer
        out = self.outconv(xd42)

        # Reshaping the output
        reshaped_output = out.view(out.size(0), self.n_classes, self.num_dates, out.size(2), out.size(3))
        softmax_output = F.softmax(reshaped_output, dim=1)
        out = softmax_output.view_as(out)
        if self.verbose:
            print("xe11", xe11.shape)
            print("xe12", xe12.shape)
            print("xp1", xp1.shape)
            print("xe21", xe21.shape)
            print("xe22", xe22.shape)
            print("xp2", xp2.shape)
            print("xe31", xe31.shape)
            print("xe32", xe32.shape)
            print("xp3", xp3.shape)
            print("xe41", xe41.shape)
            print("xe42", xe42.shape)
            print("xp4", xp4.shape)
            print("xe51", xe51.shape)
            print("xe52", xe52.shape)
            print("xu1", xu1.shape)
            print("cat U1 and e42")
            print("xu11", xu11.shape)
            print("xd11", xd11.shape)
            print("xd12", xd12.shape)
            print("xu2", xu2.shape)
            print("cat xu2 and xe32")
            print("xu22", xu22.shape)
            print("xd21", xd21.shape)
            print("xd22", xd22.shape)
            print("xu3", xu3.shape)
            print('cat xu3 and xe22')
            print("xu33", xu33.shape)
            print("xd31", xd31.shape)
            print("xd32", xd32.shape)
            print("xu4", xu4.shape)
            print("cat xu4 and xe12")
            print("xu44", xu44.shape)
            print("xd41", xd41.shape)
            print("xd42", xd42.shape)
            print("out shape", out.shape)

        return out


unet_model = UNet(n_class=3,num_dates=2, verbose=True)
print(unet_model)
unet_model.eval()
unet_input_dim = (10, 96*2, 192*2)
unstacked_unet = UnStackNetwork(unet_model, unet_input_dim)
print(unstacked_unet.output)