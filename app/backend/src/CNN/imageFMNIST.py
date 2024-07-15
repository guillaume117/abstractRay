import torch
from torchvision import datasets, transforms
import random
from PIL import Image

transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset = datasets.FashionMNIST(root='data', train=True, download=True, transform=transform)

random_index = random.randint(0, len(dataset) - 1)

image, label = dataset[random_index]

image_pil = transforms.ToPILImage()(image)

image_pil.save('test.jpeg')

