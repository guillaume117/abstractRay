import torch
import onnx 
import numpy as np
import random
import torch.nn as nn 
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure that all operations are deterministic on GPU (if applicable)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        
        self.maxpool2D = nn.MaxPool2d(kernel_size=2,stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(8,8))
        
 
        self.fc1 = nn.Linear(in_features=2048, out_features=2048)  
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=2048, out_features=128)  
        self.relu4 = nn.ReLU()

        self.fc3 = nn.Linear(in_features=128, out_features=10)  
        self.relu5 = nn.ReLU()

    def forward(self, x):

        x = self.relu1(self.conv1(x))
        

        x = self.relu2(self.conv2(x))

       

        x=self.maxpool2D(x)
        
       
        x = self.avgpool(x)
  
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
      
        x = self.relu4(self.fc2(x))
        x = self.relu5(self.fc3(x))
        return x
model = SimpleCNN()
input_dim =(3,224,224)
dummy_input = torch.randn(1, *input_dim)
torch.onnx.export(
    model,                # Le modèle PyTorch à exporter
    dummy_input,          # L'exemple d'entrée
    "model.onnx",         # Le nom du fichier de sortie ONNX
    export_params=True,   # Exporter aussi les paramètres du modèle (poids)
    opset_version=15,     # Version de l'opset d'ONNX
    do_constant_folding=True,  # Effectuer des pliages constants pour l'optimisation
    input_names=['input'],     # Nom des entrées du modèle
    output_names=['output'],   # Nom des sorties du modèle
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # Axes dynamiques (par ex. pour la taille de lot)
)
import torch
from torchvision import transforms
from PIL import Image


transform = transforms.ToPILImage()
image = transform(dummy_input.squeeze(0))


image.save("output_image.jpeg", "JPEG")

print("Image sauvegardée en format JPEG avec succès")
