import torch
import torch.nn as nn
import torchvision.models as models
import copy
import ray

import sys
sys.path.append('app/src')
sys.path.append('./src')



from sparse_evaluation_3 import SparseEvaluation  
from zono_sparse_gen import ZonoSparseGeneration
from unstack_network import UnStackNetwork
from abstract_relu import AbstractReLU
import random
import numpy as np

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
class ModelEvaluator:

    def __init__(self, unstacked_model, input,num_workers = 1,available_RAM = 8, device =torch.device('cpu')):
        self.output = unstacked_model
        self.input = input
        self.num_workers = num_workers
        self.available_RAM = available_RAM
        self.device = device
        
    def dim_chunk(self, available_RAM=None):
        if available_RAM is None:
            available_RAM = self.available_RAM
        dense_memory_footprint = torch.prod(torch.tensor(self.input.shape)) *4/1e9
        return int(max(1,available_RAM//(3*dense_memory_footprint)))
    
    
    def process_center_layer(self):

        layer = self.details['original']
        self.input = layer(self.input)


    def process_trash_layer(self):

        trash_layer = self.details[f'noise_{self.name}']
        self.trash = trash_layer(self.trash)
    


    def process_max_pool2D(self,maxpool,numworkers= None):
        dim_x = self.input.size[1]
        kernel_size = maxpool.kernel_size
        
        assert kernel_size==2,f"Maxpool2D kernel size {kernel_size}. A kernel size different of 2 is not supported"
        
        stride = maxpool.stride
        padding = maxpool.paddingkernel_size = 2





        conv_0 = nn.Conv2d(dim_x, dim_x, kernel_size=kernel_size, stride=stride, padding=padding,groups=dim_x)
        conv_1 = nn.Conv2d(dim_x, dim_x, kernel_size=kernel_size, stride=stride, padding=padding,groups=dim_x)
        conv_2 = nn.Conv2d(dim_x, dim_x, kernel_size=kernel_size, stride=stride, padding=padding,groups=dim_x)
        conv_3 = nn.Conv2d(dim_x, dim_x, kernel_size=kernel_size, stride=stride, padding=padding,groups=dim_x)
        
        w_0 = torch.tensor([[[[1., -1.], [0, 0.]]]])
        w_1 = torch.tensor([[[[0., 1.], [0, 0.]]]])
        w_2 = torch.tensor([[[[0., 0.], [0., 1.]]]])
        w_3 = torch.tensor([[[[0., 0.], [1., 0.]]]])
        
        w_0 = w_0.expand(dim_x,-1,-1,-1)
        w_1 = w_1.expand(dim_x,-1,-1,-1)
        w_2 = w_2.expand(dim_x,-1,-1,-1)
        w_3 = w_3.expand(dim_x,-1,-1,-1)
        
        conv_0.weight.data = w_0
        conv_0.bias.data =  torch.zeros(dim_x)
        conv_1.weight.data = w_1
        conv_1.bias.data =  torch.zeros(dim_x)
        conv_2.weight.data = w_2
        conv_2.bias.data =  torch.zeros(dim_x)
        conv_3.weight.data = w_3
        conv_3.bias.data =  torch.zeros(dim_x)


        c1=conv_0(self.input)
        evaluator = SparseEvaluation(self.zonotope_espilon_sparse_tensor, 
                                     chunk_size=self.dim_chunk(), 
                                     function=conv_0, 
                                     mask_coef=self.mask_epsilon, 
                                     device= self.device)
        E1, S1 = evaluator.evaluate_all_chunks(num_workers=self.num_workers)
        len_E1 = E1.size(0)

       
        
        

        x1, t1, m1= AbstractReLU.abstract_relu(
                        c1, S1, self.trash, start_index=len_E1, add_symbol=True
                    )




    def process_linear_layer(self, num_workers=None, epsilon_layer = None):

        if num_workers is None:
            num_workers=self.num_workers

       
        
        dim_chunk_val_input = self.dim_chunk()
        self.process_center_layer()
        dim_chunk_val_output = self.dim_chunk()

        dim_chunk_val = min(dim_chunk_val_input, dim_chunk_val_output)
        if epsilon_layer is None : 

            epsilon_layer = self.details[f'epsilon_{self.name}']
    
        evaluator = SparseEvaluation(self.zonotope_espilon_sparse_tensor, 
                                     chunk_size=dim_chunk_val, 
                                     function=epsilon_layer, 
                                     mask_coef=self.mask_epsilon, 
                                     device= self.device)
        
        self.zonotope_espilon_sparse_tensor, self.sum_abs = evaluator.evaluate_all_chunks(num_workers=num_workers)
        self.len_zono = self.zonotope_espilon_sparse_tensor.size(0)
        self.mask_epsilon = torch.ones_like(self.mask_epsilon)

        _, new_sparse = ZonoSparseGeneration(self.trash,from_trash=True,start_index=self.len_zono).total_zono()
        print(new_sparse)
        
        if new_sparse is not None:
            evaluator_new_noise = SparseEvaluation(new_sparse,
                                                   chunk_size = dim_chunk_val,
                                                   function=epsilon_layer, 
                                                   mask_coef = self.mask_epsilon,
                                                   eval_start_index=self.len_zono,
                                                   device =self.device)
            new_sparse, sum = evaluator_new_noise.evaluate_all_chunks(num_workers=1)
            self.sum_abs +=sum
            zono_size = list(self.zonotope_espilon_sparse_tensor.size())
            new_sparse_size = list(new_sparse.size())
            print('new s size',new_sparse_size)
            new_size  =[zono_size[0]+new_sparse_size[0],*zono_size[1:]]
            print('new z size',new_size)
            indices = torch.cat([self.zonotope_espilon_sparse_tensor.indices(), new_sparse.indices()], dim=1)
            values = torch.cat([self.zonotope_espilon_sparse_tensor.values(), new_sparse.values()])
            self.zonotope_espilon_sparse_tensor = torch.sparse_coo_tensor(indices, values, size = new_size).coalesce()
        new_sparse = None



        self.process_trash_layer()
        
        self.trash = torch.zeros_like(self.trash)


    

    
    def evaluate_model(self, zonotope_espilon_sparse_tensor, num_workers=None):

       
        if num_workers is None:
            num_workers = self.num_workers
        self.zonotope_espilon_sparse_tensor=zonotope_espilon_sparse_tensor
        results = {}
        self.mask_epsilon = torch.ones_like(self.input)
        self.trash = torch.zeros_like(self.input)



        for name, details in self.output.items():
            print(name)
            print(details)
            self.name = name
            self.details = details

            if 'original' in details:
                self.process_linear_layer()

                print('name passed = ',self.name)
                
               
            activation_name = self.details.get('activation', None)


            if activation_name:
                class_name = f"Abstract{activation_name}"
                print(f'class_name= {class_name}')
                AbstractClass = globals().get(class_name)
                if AbstractClass:
                    abstract_instance = AbstractClass()
                    self.input, self.trash, self.mask_epsilon= abstract_instance.abstract_relu(
                        self.input, self.sum_abs, self.trash, start_index=self.len_zono, add_symbol=True
                    )

                    
                    
                    

                    
        results = {
            'center': self.input,
            'min': self.input-self.sum_abs,
            'max': self.input+self.sum_abs,
            'relevance':self.zonotope_espilon_sparse_tensor
        }
        return results




class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        
 
        self.fc1 = nn.Linear(in_features=131072, out_features=128)  
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=128, out_features=10)  
        self.relu4 = nn.ReLU()

    def forward(self, x):

        x = self.relu1(self.conv1(x))

        x = self.relu2(self.conv2(x))
 

  
        x = x.view(x.size(0), -1)
    
        x = self.relu3(self.fc1(x))
      
        x = self.relu4(self.fc2(x))
        return x

"""
import onnx
from onnx2torch import convert
path = './vgg16-12.onnx'
onnx_model = onnx.load(path)
pytorch_model = convert(onnx_model)
model = pytorch_model

"""

model = SimpleCNN()
input_dim = (3,64,64
             )


unstacked = UnStackNetwork(model, input_dim)
print("*"*100)
print("unstacked output ",*unstacked.output)
print("*"*100)

test_input = torch.randn(1, *input_dim)
_,zonotope_espilon_sparse_tensor = ZonoSparseGeneration(test_input,0.001).total_zono()
print(zonotope_espilon_sparse_tensor)
ray.init()
model_evaluator = ModelEvaluator(unstacked.output, test_input,num_workers=1, available_RAM=5,device=torch.device('cpu'))

result = model_evaluator.evaluate_model(zonotope_espilon_sparse_tensor)

print(f"True:{model(test_input)}")
print(f"Center: {result['center']}")
print(f"Min: {result['min']}")
print(f"Max: {result['max']}")
print(f"Relevance: {result['relevance']}")
