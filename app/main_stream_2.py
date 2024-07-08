from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.gzip import GZipMiddleware
import torch
import onnx
from onnx2torch import convert
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import ray
import os
import sys
sys.path.append('app/src')
sys.path.append('./src')
from util import sparse_tensor_stats, resize_sparse_coo_tensor
from zono_sparse_gen import ZonoSparseGeneration
from model_evaluator import ModelEvaluator
from unstack_network2 import UnStackNetwork
import io
import uvicorn

app = FastAPI()

# Middleware to increase max file size
app.add_middleware(
    GZipMiddleware,
    minimum_size=1000
)

# Global variables to store intermediate results
intermediate_results = {}

def load_model(network_file, model_name):
    if model_name == "custom":
        filename = network_file.filename
        if filename.endswith('.onnx'):
            onnx_model = onnx.load_model(io.BytesIO(network_file.file.read()))
            pytorch_model = convert(onnx_model)
            pytorch_model.eval()
            return pytorch_model
        elif filename.endswith('.pt') or filename.endswith('.pth'):
            pytorch_model = torch.load(io.BytesIO(network_file.file.read()), map_location=torch.device('cpu'))
            pytorch_model.eval()
            return pytorch_model
        else:
            raise ValueError("Unsupported network model format. Please provide a .onnx or .pt/.pth file.")
    elif model_name == "vgg16":
        return models.vgg16(pretrained=True).eval()
    elif model_name == "vgg19":
        return models.vgg19(pretrained=True).eval()
    elif model_name == "resnet":
        return models.resnet18(pretrained=True).eval()
    else:
        raise ValueError("Unsupported model name.")

def validate_and_transform_image(image_data, resize_input, resize_width, resize_height):
    image = Image.open(io.BytesIO(image_data))
    if resize_input:
        image = image.resize((resize_width, resize_height))
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    transformed_image = transform(image)
    return transformed_image, image.size

@app.post("/prepare_evaluation/")
async def prepare_evaluation(
    network: UploadFile = File(None),
    input_image: UploadFile = File(...),
    model_name: str = Form(...),
    num_worker: int = Form(...),
    back_end: str = Form(...),
    num_symbol: str = Form(...),
    noise: float = Form(...),
    RAM: float = Form(...),
    resize_input: bool = Form(...),
    resize_width: int = Form(None),
    resize_height: int = Form(None)
):
    try:
        # Supprimer le fichier de signal d'interruption s'il existe
        if os.path.exists('interrupt_signal'):
            os.remove('interrupt_signal')

        messages = []

        # Charger le modèle
        model = load_model(network, model_name)
        messages.append("load_model ok")

        # Charger et transformer l'image
        image_data = await input_image.read()
        image_tensor, image_size = validate_and_transform_image(image_data, resize_input, resize_width, resize_height)
        messages.append(f"Image loaded successfully, size: {image_size[0]}x{image_size[1]} pixels")

        if image_tensor is not None:
            image_tensor = image_tensor.unsqueeze(0)
            model.eval()
            with torch.no_grad():
                output = model(image_tensor)

            # Générer le zonotope
            _, zonotope_espilon_sparse_tensor = ZonoSparseGeneration(image_tensor, noise).total_zono()
            messages.append(f"Zonotope generated successfully, dimensions: {zonotope_espilon_sparse_tensor.shape}")

            if num_symbol != 'full':
                try:
                    num_symbol = int(num_symbol)
                    zonotope_espilon_sparse_tensor = resize_sparse_coo_tensor(zonotope_espilon_sparse_tensor, (num_symbol, *image_tensor.shape[1:]))
                except ValueError:
                    raise HTTPException(status_code=400, detail="Invalid value for num_symbol. It should be 'full' or an integer.")

            # Exécuter UnStackNetwork
            unstack_network = UnStackNetwork(model, image_tensor.shape[1:])
            messages.append("UnStackNetwork executed successfully")

            # Stocker les résultats intermédiaires
            global intermediate_results
            intermediate_results = {
                'model': model,
                'image_tensor': image_tensor,
                'zonotope_espilon_sparse_tensor': zonotope_espilon_sparse_tensor,
                'num_worker': num_worker,
                'back_end': back_end,
                'RAM': RAM,
                'unstack_network': unstack_network
            }

            response = {
                "messages": messages
            }
            return JSONResponse(content=response)
        else:
            raise HTTPException(status_code=400, detail="Failed to process the image.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/execute_evaluation/")
async def execute_evaluation():
    try:
        global intermediate_results
        if not intermediate_results:
            raise HTTPException(status_code=400, detail="No intermediate results found. Please run prepare_evaluation first.")

        model = intermediate_results['model']
        image_tensor = intermediate_results['image_tensor']
        zonotope_espilon_sparse_tensor = intermediate_results['zonotope_espilon_sparse_tensor']
        num_worker = intermediate_results['num_worker']
        back_end = intermediate_results['back_end']
        RAM = intermediate_results['RAM']
        unstack_network = intermediate_results['unstack_network']

        if back_end == 'cuda':
            sparse_worker_decorator = ray.remote(num_gpus=1)
        else:
            sparse_worker_decorator = ray.remote(num_cpus=1)

        model_evaluator = ModelEvaluator(
            unstack_network.output,
            image_tensor,
            num_workers=num_worker,
            available_RAM=RAM,
            device=torch.device(back_end)
        )

        # Vérifiez l'interruption périodiquement
        for i in range(10):  # Simulation d'une évaluation itérative
            if os.path.exists('interrupt_signal'):
                raise HTTPException(status_code=400, detail="Evaluation interrupted by the user.")
            # Ajoutez votre logique d'évaluation ici

        # Évaluation du modèle
        result = model_evaluator.evaluate_model(zonotope_espilon_sparse_tensor)
        argmax = torch.topk(model(image_tensor).squeeze(0), 10).indices

        response = {
            "argmax": argmax.tolist(),
            "true": model(image_tensor).squeeze(0)[argmax].tolist(),
            "center": result['center'].squeeze(0)[argmax].tolist(),
            "min": result['min'].squeeze(0)[argmax].tolist(),
            "max": result['max'].squeeze(0)[argmax].tolist(),
            "diff_center_true": torch.max(model(image_tensor) - result['center']).item()
        }
        return JSONResponse(content=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/interrupt_evaluation/")
async def interrupt_evaluation():
    with open('interrupt_signal', 'w') as f:
        f.write('interrupt')
    return {"detail": "Evaluation interruption signal sent"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
