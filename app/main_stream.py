from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
import onnx
from onnx2torch import convert
from PIL import Image
import torchvision.transforms as transforms
import ray
import sys 
sys.path.append('app/src')
sys.path.append('./src')
from util import sparse_tensor_stats, resize_sparse_coo_tensor
from zono_sparse_gen import ZonoSparseGeneration
from model_evaluator import ModelEvaluator
from unstack_network2 import UnStackNetwork
import io

app = FastAPI()

def load_model(network_path):
    if network_path.endswith('.onnx'):
        onnx_model = onnx.load(network_path)
        pytorch_model = convert(onnx_model)
        pytorch_model.eval()
        return pytorch_model
    elif network_path.endswith('.pt') or network_path.endswith('.pth'):
        pytorch_model = torch.load(network_path)
        pytorch_model.eval()
        return pytorch_model
    else:
        raise ValueError("Unsupported network model format. Please provide a .onnx or .pt/.pth file.")

def validate_and_transform_image(image_data):
    image = Image.open(io.BytesIO(image_data))
    image_size = image.size
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])
    transformed_image = transform(image)
    return transformed_image

@app.post("/evaluate_model/")
async def evaluate_model(
    network: UploadFile = File(...),
    input_image: UploadFile = File(...),
    num_worker: int = 1,
    back_end: str = 'cpu',
    num_symbol: str = 'full',
    noise: float = 0.0,
    RAM: float = 1.0
):
    try:
        model = load_model(network.file)
        image_data = await input_image.read()
        image_tensor = validate_and_transform_image(image_data)

        if image_tensor is not None:
            image_tensor = image_tensor.unsqueeze(0)
            model.eval()
            with torch.no_grad():
                output = model(image_tensor)

            _, zonotope_espilon_sparse_tensor = ZonoSparseGeneration(image_tensor, noise).total_zono()

            if num_symbol != 'full':
                zonotope_espilon_sparse_tensor = resize_sparse_coo_tensor(zonotope_espilon_sparse_tensor, (int(num_symbol), *image_tensor.shape[1:]))

            if back_end == 'cuda':
                sparse_worker_decorator = ray.remote(num_gpus=1)
            else:
                sparse_worker_decorator = ray.remote(num_cpus=1)

            model_evaluator = ModelEvaluator(
                UnStackNetwork(model, image_tensor.shape[1:]).output,
                image_tensor,
                num_workers=num_worker,
                available_RAM=RAM,
                device=torch.device(back_end)
            )

            result = model_evaluator.evaluate_model(zonotope_espilon_sparse_tensor)
            argmax = torch.topk(result['center'], 10).indices
            response = {
                "argmax": argmax.tolist(),
                "true": model(image_tensor).squeeze(0)[argmax].tolist(),
                "center": result['center'].squeeze(0)[argmax].tolist(),
                "min": result['min'].squeeze(0)[argmax].tolist(),
                "max": result['max'].squeeze(0)[argmax].tolist(),
                "diff_center_true": torch.max(model(image_tensor) - result['center']).item()
            }
            return JSONResponse(content=response)
        else:
            raise HTTPException(status_code=400, detail="Failed to process the image.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
