from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.gzip import GZipMiddleware
import torch
import onnx
from onnx2torch import convert
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import os
import io
import uvicorn
from AbstractRay.backend.src.util import SimpleCNN
from AbstractRay.backend.src.zono_sparse_gen import ZonoSparseGeneration
from AbstractRay.backend.src.evaluator import ModelEvaluator
from AbstractRay.backend.src.unstack_network import UnStackNetwork

os.environ["RAY_NUM_CPUS"] = str(os.cpu_count())

backend_url = os.getenv("BACKEND_URL", "http://abstratray:8000")
app = FastAPI()

app.add_middleware(
    GZipMiddleware,
    minimum_size=1000
)

intermediate_results = {}

def load_model(network_file, network_name):
    """
    Load a model based on the provided network file or predefined network name.

    Args:
        network_file (UploadFile): The uploaded network file.
        network_name (str): The name of the predefined network.

    Returns:
        torch.nn.Module: The loaded PyTorch model.

    Raises:
        ValueError: If the network model format is unsupported or the model name is unsupported.
    """
    if network_name == "custom":
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
    elif network_name == "vgg16":
        return models.vgg16(pretrained=True).eval()
    elif network_name == "vgg19":
        return models.vgg19(pretrained=True).eval()
    elif network_name == "resnet":
        return models.resnet18(pretrained=True).eval()
    elif network_name == "simplecnn":
        model = SimpleCNN()
        model_weights_path = './src/CNN/simple_cnn_fashionmnist.pth'
        model.load_state_dict(torch.load(model_weights_path))
        model.eval()
        return model
    else:
        raise ValueError("Unsupported model name.")

def validate_and_transform_image(image_data, resize_input, resize_width, resize_height):
    """
    Validate and transform the input image.

    Args:
        image_data (bytes): The input image data.
        resize_input (bool): Whether to resize the input image.
        resize_width (int): The width to resize the image to.
        resize_height (int): The height to resize the image to.

    Returns:
        torch.Tensor: The transformed image tensor.
        tuple: The size of the original image.
    """
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
    network_name: str = Form(...),
    num_worker: int = Form(...),
    back_end: str = Form(...),
    noise: float = Form(...),
    RAM: float = Form(...),
    resize_input: bool = Form(...),
    resize_width: int = Form(None),
    resize_height: int = Form(None),
    box_input: bool = Form(...),
    box_x_min: int = Form(None),
    box_x_max: int = Form(None),
    box_y_min: int = Form(None),
    box_y_max: int = Form(None),
    add_symbol: bool = Form(...),
    relevance_dump: bool = Form(...),
    model_last_layer:int = Form(None)
):
    """
    Prepare the evaluation by loading the model, processing the input image, and generating the zonotope.

    Args:
        network (UploadFile, optional): The uploaded network file.
        input_image (UploadFile): The uploaded input image file.
        network_name (str): The name of the network to load.
        num_worker (int): The number of workers for processing.
        back_end (str): The backend device for processing.
        noise (float): The noise level for the evaluation.
        RAM (float): The available RAM for the evaluation.
        resize_input (bool): Whether to resize the input image.
        resize_width (int, optional): The width to resize the image to.
        resize_height (int, optional): The height to resize the image to.
        box_input (bool): Whether to use a masked region of the image.
        box_x_min (int, optional): The minimum x-coordinate of the mask.
        box_x_max (int, optional): The maximum x-coordinate of the mask.
        box_y_min (int, optional): The minimum y-coordinate of the mask.
        box_y_max (int, optional): The maximum y-coordinate of the mask.
        add_symbol (bool): Whether to add a symbol during evaluation.
        relevance_dump (bool): Whether to renew the abstract domain.
        model_last_layer (int,optional): For evaluate only the first args layers.
    Returns:
        JSONResponse: The response containing the messages and status of the preparation.
    """
    try:
        if os.path.exists('interrupt_signal'):
            os.remove('interrupt_signal')

        messages = []

        model = load_model(network, network_name)
        messages.append("load_model ok")

        image_data = await input_image.read()
        image_tensor, image_size = validate_and_transform_image(image_data, resize_input, resize_width, resize_height)
        messages.append(f"Image loaded successfully, size: {image_size[0]}x{image_size[1]} pixels")

        if image_tensor is not None:
            image_tensor = image_tensor.unsqueeze(0)
            model.eval()
            with torch.no_grad():
                output = model(image_tensor)

            if box_input == False:
                zonotope_espilon_sparse_tensor = ZonoSparseGeneration().zono_from_noise_level_and_tensor(noise_level=noise, tensor=image_tensor)
                messages.append(f"Zonotope generated successfully, dimensions: {zonotope_espilon_sparse_tensor.shape}")
            else:
                zonotope_espilon_sparse_tensor = ZonoSparseGeneration().zono_from_input_noise_level_and_mask(
                    tensor_input=image_tensor,
                    x_min=box_x_min,
                    x_max=box_x_max,
                    y_min=box_y_min,
                    y_max=box_y_max,
                    noise_level=noise
                )
            unstack_network = UnStackNetwork(model, image_tensor.shape[1:],last_layer=model_last_layer)
            messages.append("UnStackNetwork executed successfully")

            global intermediate_results
            intermediate_results = {
                'model': model,
                'network_name': network_name,
                'image_tensor': image_tensor,
                'zonotope_espilon_sparse_tensor': zonotope_espilon_sparse_tensor,
                'num_worker': num_worker,
                'back_end': back_end,
                'RAM': RAM,
                'unstack_network': unstack_network,
                'add_symbol': add_symbol,
                'renew_abstract_domain': relevance_dump,
                'noise_level': noise,
                'model_last_layer':model_last_layer

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
    """
    Execute the evaluation using the prepared intermediate results.

    Returns:
        JSONResponse: The response containing the evaluation results.
    """
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
        add_symbol = intermediate_results['add_symbol']
        network_name = intermediate_results['network_name']
        noise_level = intermediate_results['noise_level']
        os.environ["RAY_BACKEND"]=back_end
        abstract_domain = {
            'zonotope': zonotope_espilon_sparse_tensor,
            'center': image_tensor,
            'sum': torch.zeros_like(image_tensor),
            'trash': torch.zeros_like(image_tensor),
            'mask': torch.ones_like(image_tensor),
            'perfect_domain': True
        }

        model_evaluator = ModelEvaluator(
            unstack_network.output,
            abstract_domain,
            num_workers=num_worker,
            available_RAM=RAM,
            device=torch.device(back_end),
            add_symbol=add_symbol,
            json_file_prefix=str(network_name),
            noise_level=noise_level
        )

        for i in range(10):
            if os.path.exists('interrupt_signal'):
                raise HTTPException(status_code=400, detail="Evaluation interrupted by the user.")

        abstract_domain = model_evaluator.evaluate_model()
        argmax = torch.topk(model(image_tensor).squeeze(0), 10).indices
        size = image_tensor.numel()

        response = {
            "argmax": argmax.tolist(),
            "true": model(image_tensor).squeeze(0)[argmax].tolist(),
            "center": abstract_domain['center'].squeeze(0)[argmax].tolist(),
            "min": (abstract_domain['center'].squeeze(0)[argmax] - abstract_domain['sum'].squeeze(0)[argmax]).tolist(),
            "max": (abstract_domain['center'].squeeze(0)[argmax] + abstract_domain['sum'].squeeze(0)[argmax]).tolist(),
            "diff_center_true": torch.max(model(image_tensor) - abstract_domain['center']).item(),
        }
        return JSONResponse(content=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/interrupt_evaluation/")
async def interrupt_evaluation():
    """
    Interrupt the evaluation process by creating an interrupt signal file.

    Returns:
        dict: A dictionary indicating that the interruption signal was sent.
    """
    with open('interrupt_signal', 'w') as f:
        f.write('interrupt')
    return {"detail": "Evaluation interruption signal sent"}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
