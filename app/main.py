import argparse
import torch
import onnx
from onnx2torch import convert
from PIL import Image
import torchvision.transforms as transforms
import sys 
import os
sys.path.append('./app')
sys.path.append('./app/backend')
sys.path.append('./app/backend/src')
sys.path.append('./app/backend/src/cpuconv2D')
import ray 
from util import sparse_tensor_stats , resize_sparse_coo_tensor,ensure_ray_initialized
from zono_sparse_gen import ZonoSparseGeneration
from model_evaluator_refacto import ModelEvaluator
from unstack_network2 import UnStackNetwork
os.environ["RAY_NUM_CPUS"] = str(os.cpu_count())

ensure_ray_initialized()

def parse_args():
    parser = argparse.ArgumentParser(description='Argument Parser for Network Configuration')
    
    parser.add_argument('--network', type=str, required=True, help='Path to the network model (ONNX or PyTorch)')
    parser.add_argument('--input', type=str, required=True, help='Input image path')
    parser.add_argument('--num_worker', type=int, required=True, help='Number of workers')
    parser.add_argument('--back_end', choices=['cuda', 'cpu'], required=True, help='Backend to use: cuda or cpu')
    parser.add_argument('--num_symbol', required=True, help='Number of symbols: \'full\' or a positive integer')
    parser.add_argument('--noise', type=float, required=True, help='Noise level')
    parser.add_argument('--RAM',type = float, required = True, help ='Available RAM per worker')

    args = parser.parse_args()
    
    model = None

    # Vérification et conversion du modèle ONNX ou chargement du modèle PyTorch
    if args.network.endswith('.onnx'):
        try:
            onnx_model = onnx.load(args.network)
            pytorch_model = convert(onnx_model)
            model = pytorch_model
            model.eval()
            print("ONNX model successfully converted to PyTorch.")
        except Exception as e:
            print(f"Error converting ONNX model: {e}")
    elif args.network.endswith('.pt') or args.network.endswith('.pth'):
        try:
            model = torch.load(args.network)
            model.eval()
            print("PyTorch model successfully loaded.")
        except Exception as e:
            print(f"Error loading PyTorch model: {e}")
            
    else:
        print("Unsupported network model format. Please provide a .onnx or .pt/.pth file.")
    
    return args, model

def validate_and_transform_image(image_path):
    try:
        image = Image.open(image_path)
        image_size = image.size
        print(f"Image size: {image_size}")
        
        transform = transforms.Compose([
            transforms.Resize(image_size),  # You can change the size according to your model's requirement
            transforms.ToTensor(),
        ])
        
        transformed_image = transform(image)
        return transformed_image
    except Exception as e:
        print(f"Error processing the image: {e}")
        return None

if __name__ == '__main__':
    args, model = parse_args()
    
    if model:
        image_tensor = validate_and_transform_image(args.input)
        
        if image_tensor is not None:
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
            try:
                model.eval()
                with torch.no_grad():
                    output = model(image_tensor)
                print("Model inference successful")
                _,zonotope_espilon_sparse_tensor = ZonoSparseGeneration(image_tensor,args.noise).total_zono()
                if args.num_symbol !='full':
                    zonotope_espilon_sparse_tensor = resize_sparse_coo_tensor(zonotope_espilon_sparse_tensor, (int(args.num_symbol),*image_tensor.shape[1:]))
                print(f"Abstract domain input dim = {zonotope_espilon_sparse_tensor.size()}, noise = {args.noise}")
                unstacked = UnStackNetwork(model, image_tensor.shape[1:])
                print("*"*100)
                print("unstacked output ",*unstacked.output)
                print("*"*100)
                if args.back_end == 'cuda':
                    sparse_worker_decorator = ray.remote(num_gpus=1)
                else:
                    sparse_worker_decorator = ray.remote(num_cpus=1)

                abstract_domain = {
                    'zonotope' : zonotope_espilon_sparse_tensor,
                    'center' : image_tensor,
                    'sum': torch.zeros_like(image_tensor),
                    'trash': torch.zeros_like(image_tensor),
                    'mask': torch.ones_like(image_tensor),
                    'perfect_domain':True


                }
                model_evaluator = ModelEvaluator(unstacked.output, abstract_domain,num_workers=args.num_worker, available_RAM=args.RAM,device=torch.device(args.back_end))

                result = model_evaluator.evaluate_model()                
            except Exception as e:
                print(f"Error during model inference: {e}")
        else:
            print("Failed to process the image.")
    else:
        print("Failed to load the model.")



