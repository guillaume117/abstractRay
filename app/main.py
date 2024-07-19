import argparse
import os
import io
import torch
import onnx
from onnx2torch import convert
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models


from app.backend.src.util import  SimpleCNN, ensure_ray_initialized
from app.backend.src.zono_sparse_gen import ZonoSparseGeneration
from app.backend.src.evaluator import ModelEvaluator
from app.backend.src.unstack_network import UnStackNetwork

os.environ["RAY_NUM_CPUS"] = str(os.cpu_count())


def load_model(network_name,network_file=None):
    if network_name == "custom":
        if network_file.endswith('.onnx'):
            onnx_model = onnx.load_model(network_file)
            pytorch_model = convert(onnx_model)
            pytorch_model.eval()
            return pytorch_model
        elif network_file.endswith('.pt') or network_file.endswith('.pth'):
            pytorch_model = torch.load(network_file, map_location=torch.device('cpu'))
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
        model_weights_path = './app/backend/src/CNN/simple_cnn_fashionmnist.pth'
        model.load_state_dict(torch.load(model_weights_path))
        model.eval()
        return model
    
    else:
        print(network_name)
        raise ValueError("Unsupported model name.")

def validate_and_transform_image(image_path, resize_input=False, resize_width=None, resize_height=None):
    image = Image.open(image_path)
    if resize_input:
        pass
        #image = image.resize((resize_width, resize_height))
    transform = transforms.Compose([transforms.ToTensor()])
    transformed_image = transform(image)
    return transformed_image, image.size

def main(args):
    try:
        messages = []

        # Charger le modèle
        model = load_model(args.network_name, args.network_file)
        messages.append("load_model ok")
        print(messages)

        # Charger et transformer l'image
        image_tensor, image_size = validate_and_transform_image(args.input_image, args.resize_input, args.resize_width, args.resize_height)
        messages.append(f"Image loaded successfully, size: {image_size[0]}x{image_size[1]} pixels")
        print(messages)
        if image_tensor is not None:
            image_tensor = image_tensor.unsqueeze(0)
            model.eval()
            with torch.no_grad():
                output = model(image_tensor)

            # Générer le zonotope
            if not args.box_input:
                zonotope_espilon_sparse_tensor = ZonoSparseGeneration().zono_from_noise_level_and_tensor(noise_level=args.noise, tensor=image_tensor)
                messages.append(f"Zonotope generated successfully, dimensions: {zonotope_espilon_sparse_tensor.shape}")
            else:
                zonotope_espilon_sparse_tensor = ZonoSparseGeneration().zono_from_input_noise_level_and_mask(
                    tensor_input=image_tensor,
                    x_min=args.box_x_min,
                    x_max=args.box_x_max,
                    y_min=args.box_y_min,
                    y_max=args.box_y_max,
                    noise_level=args.noise
                )

            # Exécuter UnStackNetwork
            unstack_network = UnStackNetwork(model, image_tensor.shape[1:])
            messages.append("UnStackNetwork executed successfully")

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
                num_workers=args.num_worker,
                available_RAM=args.RAM,
                device=torch.device(args.back_end),
                add_symbol=args.add_symbol,
                json_file_prefix=str(args.network_name),
                noise_level=args.noise
            )

            abstract_domain = model_evaluator.evaluate_model()
            argmax = torch.topk(model(image_tensor).squeeze(0), 10).indices

            response = {
                "argmax": argmax.tolist(),
                "true": model(image_tensor).squeeze(0)[argmax].tolist(),
                "center": abstract_domain['center'].squeeze(0)[argmax].tolist(),
                "min": (abstract_domain['center'].squeeze(0)[argmax] - abstract_domain['sum'].squeeze(0)[argmax]).tolist(),
                "max": (abstract_domain['center'].squeeze(0)[argmax] + abstract_domain['sum'].squeeze(0)[argmax]).tolist(),
                "diff_center_true": torch.max(model(image_tensor) - abstract_domain['center']).item()
            }

            print("Evaluation Results:")
            for key, value in response.items():
                print(f"{key}: {value}")
        else:
            raise ValueError("Failed to process the image.")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model evaluations.")
    parser.add_argument("--network_file", type=str, required=False, help="Path to the network file (.onnx, .pt, .pth).")
    parser.add_argument("--input_image", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--network_name", type=str, required=True, choices=["custom", "vgg16", "vgg19", "resnet", "simplecnn"], help="Name of the network.")
    parser.add_argument("--num_worker", type=int, required=True, help="Number of workers.")
    parser.add_argument("--back_end", type=str, required=True, choices=["cpu", "cuda"], help="Backend device.")
    parser.add_argument("--noise", type=float, required=True, help="Noise level.")
    parser.add_argument("--RAM", type=float, required=True, help="Available RAM.")
    parser.add_argument("--resize_input", type=bool, required=True, help="Whether to resize the input image.")
    parser.add_argument("--resize_width", type=int, help="Width to resize the input image to.")
    parser.add_argument("--resize_height", type=int, help="Height to resize the input image to.")
    parser.add_argument("--box_input", type=bool, required=True, help="Whether to use a bounding box for the input.")
    parser.add_argument("--box_x_min", type=int, help="Minimum x value for the bounding box.")
    parser.add_argument("--box_x_max", type=int, help="Maximum x value for the bounding box.")
    parser.add_argument("--box_y_min", type=int, help="Minimum y value for the bounding box.")
    parser.add_argument("--box_y_max", type=int, help="Maximum y value for the bounding box.")
    parser.add_argument("--add_symbol", type=bool, required=True, help="Whether to add symbols.")
    parser.add_argument("--relevance_dump", type=bool, required=True, help="Whether to dump relevance information.")

    args = parser.parse_args()
    main(args)
