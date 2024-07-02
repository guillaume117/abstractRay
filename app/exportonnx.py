import torch
import onnxruntime as ort
import numpy as np
import random
import torch.nn as nn
from torchvision import transforms
from PIL import Image

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
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
        self.maxpool2D = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(8, 8))
        self.fc1 = nn.Linear(in_features=2048, out_features=512)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=512, out_features=10)
        self.relu4 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.maxpool2D(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        return x

model = SimpleCNN()
input_dim = (3, 112, 112)
dummy_input = torch.randn(1, *input_dim)
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=15,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)


transform = transforms.ToPILImage()
image = transform(dummy_input.squeeze(0))
image.save("output_image.jpeg", "JPEG")
print("Image sauvegardée en format JPEG avec succès")



image_path = "output_image.jpeg"
image = Image.open(image_path)


transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
])

input_tensor = transform(image).unsqueeze(0) 



ort_session = ort.InferenceSession("model.onnx")


ort_inputs = {ort_session.get_inputs()[0].name: input_tensor.numpy()}


ort_outs = ort_session.run(None, ort_inputs)

print("ONNX Runtime output:", ort_outs)

model.eval()
with torch.no_grad():
    torch_out = model(input_tensor)

print("PyTorch model output:", torch_out)

np.testing.assert_allclose(torch_out.numpy(), ort_outs[0], rtol=1e-03, atol=1e-05)
print("The outputs from PyTorch and ONNX Runtime are close enough!")

def load_image(image_path, input_size):
    """
    Charge et redimensionne une image.

    Args:
        image_path (str): Le chemin vers l'image.
        input_size (tuple): La taille à laquelle redimensionner l'image (H, W).

    Returns:
        np.ndarray: L'image redimensionnée.
    """
   

   
    image = Image.open(image_path)


    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
    ])
    image = transform(image)
    return image

def add_noise(image, alpha):
    """
    Ajoute un bruit aléatoire indépendant à chaque pixel de l'image.

    Args:
        image (np.ndarray): L'image d'entrée.
        alpha (float): L'intensité du bruit.

    Returns:
        np.ndarray: L'image avec le bruit ajouté.
    """
    noise = np.random.uniform(-alpha, alpha, image.shape).astype(np.float32)
    noisy_image = image + noise

    return noisy_image

def estimate_min_max(model_path, image_path, input_size, alpha, num_samples=1000):
    """
    Estime les valeurs minimales et maximales de la sortie du modèle ONNX pour une image
    avec du bruit ajouté.

    Args:
        model_path (str): Le chemin vers le modèle ONNX.
        image_path (str): Le chemin vers l'image.
        input_size (tuple): La taille à laquelle redimensionner l'image (H, W).
        alpha (float): L'intensité du bruit.
        num_samples (int): Le nombre d'échantillons à générer.

    Returns:
        tuple: Les valeurs minimale et maximale estimées.
    """

    ort_session = ort.InferenceSession(model_path)

    image = load_image(image_path, input_size)
    input_name = ort_session.get_inputs()[0].name

    outputs = []
    for _ in range(num_samples):
        noisy_image = add_noise(image, alpha)
        noisy_image = np.expand_dims(noisy_image, axis=0).astype(np.float32)  
        output = ort_session.run(None, {input_name: noisy_image})
        outputs.append(output)

  
    min_output = np.min(outputs,axis=0)
    max_output = np.max(outputs,axis=0)

    return min_output, max_output

if __name__ == "__main__":
    model_path = "model.onnx"  
    image_path = "output_image.jpeg"   
    input_size = (3,112, 112)                
    alpha = 0.0001 *255                            
    num_samples = 100000                    

    min_output, max_output = estimate_min_max(model_path, image_path, input_size, alpha, num_samples)
    print(f"Valeur minimale estimée: {min_output}")
    print(f"Valeur maximale estimée: {max_output}")
