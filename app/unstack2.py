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
            x = self.process_layer(name, module, x)
        
        # Ajouter une couche Linear identité à la fin du réseau
        x = self.add_identity_layer(x)

    def process_layer(self, name, layer, x):
        if isinstance(layer, nn.Linear):
            x = nn.Flatten()(x)
            x = self.process_linear_layer(name, nn.Sequential(nn.Flatten(), layer), x)
        elif isinstance(layer, nn.Conv2d):
            x = self.process_conv_layer(name, layer, x)
        elif isinstance(layer, nn.AdaptiveAvgPool2d):
            x = self.process_avgpool_layer(name, layer, x)
        elif isinstance(layer, (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.MaxPool2d)):
            x = self.process_activation_layer(name, layer, x)
        elif isinstance(layer, nn.Flatten):
            x = self.process_flatten(name, layer, x)
        elif isinstance(layer, nn.Sequential):
            x = self.process_sequential(name, layer, x)
        else:
            if layer is not None and not 'Module()' and not 'OnnxDropoutDynamic()':
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
    
    def process_avgpool_layer(self, name, layer, x):
        self.output[name] = {
            'type': type(layer),
            'original': copy.deepcopy(layer),
            'epsilon_{}'.format(name): copy.deepcopy(layer),
            'noise_{}'.format(name): copy.deepcopy(layer),
            'output_dim': self.compute_output_dim(layer, x)
        }
        return layer(x)

    def process_activation_layer(self, name, layer, x):
        self.output[name] = {
            'activation': layer.__class__.__name__,
            'activation_function': layer
        }
        return layer(x)

    def process_sequential(self, name, layer, x):
        for sub_name, sub_layer in layer.named_children():
            x = self.process_layer(f"{name}_{sub_name}", sub_layer, x)
        return x

    def process_concat_layer(self, name, layer, x1, x2):
        concatenated = torch.cat((x1, x2), dim=1)
        self.output[name] = {
            'type': 'Concat',
            'output_dim': concatenated.shape
        }
        return concatenated

    def add_identity_layer(self, x):
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

# Exécution de test avec un modèle U-Net ou similaire peut être ajoutée ici pour vérifier le comportement du script
import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Encoder
        self.encoder1 = self.conv_block(1, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)

        # Bottleneck
        self.bottleneck = self.conv_block(256, 512)

        # Decoder
        self.upconv3 = self.upconv_block(512, 256)
        self.decoder3 = self.conv_block(512, 256)
        self.upconv2 = self.upconv_block(256, 128)
        self.decoder2 = self.conv_block(256, 128)
        self.upconv1 = self.upconv_block(128, 64)
        self.decoder1 = self.conv_block(128, 64)

        # Final output layer
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(nn.MaxPool2d(2)(enc1))
        enc3 = self.encoder3(nn.MaxPool2d(2)(enc2))

        # Bottleneck
        bottleneck = self.bottleneck(nn.MaxPool2d(2)(enc3))

        # Decoder
        dec3 = self.upconv3(bottleneck)
        dec3 = self.decoder3(torch.cat((dec3, enc3), dim=1))
        dec2 = self.upconv2(dec3)
        dec2 = self.decoder2(torch.cat((dec2, enc2), dim=1))
        dec1 = self.upconv1(dec2)
        dec1 = self.decoder1(torch.cat((dec1, enc1), dim=1))

        # Final output layer
        return self.final_conv(dec1)

# Example usage
model = UNet()
input_tensor = torch.randn(1, 1, 256, 256)  # Batch size 1, 1 input channel, 256x256 image
output = model(input_tensor)
print(output.shape)  # Should print torch.Size([1, 1, 256, 256])
unstack = UnStackNetwork(model = model, input_dim=(1,256,256))
print(unstack.output)