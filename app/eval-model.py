import torch
import torch.nn as nn
import torchvision.models as models
import copy
import ray

import sys
sys.path.append('app/src')
sys.path.append('./src')


from sparse_evaluation_2 import SparseEvaluation  
from zono_sparse_gen import ZonoSparseGeneration
from unstack_network import UnStackNetwork
from abstract_relu import AbstractReLU


class LayerEvaluator:

    def __init__(self, output, input):
        self.output = output
        self.input = input

        
    def dim_chunk(self, available_RAM):
        dense_memory_footprint = torch.prod(torch.tensor(self.input.shape)) *4/1e9
        return int(max(1,available_RAM//(5*dense_memory_footprint)))
    

    def process_linear_layer(self, name, num_workers=1):
        self.mask_epsilon = torch.ones_like(self.input)
       
        layer = self.details['original']
        dim_chunk_val_input = self.dim_chunk(available_RAM=8)
        print(dim_chunk_val_input)
        self.input = layer(self.input)
        print("self.input.size after lin",self.input.size())

        dim_chunk_val_output = self.dim_chunk(available_RAM=8)
        print(dim_chunk_val_output)
        dim_chunk_val = min(dim_chunk_val_input, dim_chunk_val_output)
        epsilon_layer = self.details[f'epsilon_{name}']
        print("espislon layer",epsilon_layer)

        evaluator = SparseEvaluation(self.zonotope_espilon_sparse_tensor, chunk_size=dim_chunk_val, function=epsilon_layer, mask_coef=self.mask_epsilon)
        self.zonotope_espilon_sparse_tensor, self.sum_abs = evaluator.evaluate_all_chunks(num_workers=num_workers)
        self.len_zono = self.zonotope_espilon_sparse_tensor.size(0)

        trash_layer = self.details[f'noise_{name}']
        
        self.trash = trash_layer(self.trash)
    
    def evaluate_layers(self, zonotope_espilon_sparse_tensor, num_workers=9):
        self.zonotope_espilon_sparse_tensor=zonotope_espilon_sparse_tensor
        results = {}
        self.mask_epsilon = torch.ones_like(self.input)
        self.trash = torch.zeros_like(self.input)
        for name, details in self.output.items():
            print(name)
            print(details)
            self.details = details

            if 'original' in details:
                self.process_linear_layer(name,num_workers=num_workers)
                print('name passed = ',name)
                
                """
                layer = details['original']
                dim_chunk_val_0 = self.dim_chunk(available_RAM=8)
                print(dim_chunk_val_0)
                self.input = layer(self.input)

                dim_chunk_val_1 = self.dim_chunk(available_RAM=8)
                print(dim_chunk_val_1)
                dim_chunk_val = min(dim_chunk_val_0,dim_chunk_val_1)
                epsilon_layer = details[f'epsilon_{name}']
            
                
                evaluator = SparseEvaluation(zonotope_espilon_sparse_tensor, chunk_size=dim_chunk_val, function=epsilon_layer, mask_coef=mask_epsilon)
                zonotope_espilon_sparse_tensor, sum_abs = evaluator.evaluate_all_chunks(num_workers=num_workers)
                start_index = zonotope_espilon_sparse_tensor.size(0)
                
                trash_layer = details[f'noise_{name}']
                trash  = trash_layer(trash)
                
                """
            activation_name = details.get('activation', None)


            if activation_name:
                class_name = f"Abstract{activation_name}"
                print(f'class_name= {class_name}')
                AbstractClass = globals().get(class_name)
                if AbstractClass:
                    abstract_instance = AbstractClass()
                    self.input, self.trash, self.mask_epsilon, new_sparse = abstract_instance.abstract_relu(
                        self.input, self.sum_abs, self.trash, start_index=self.len_zono, add_symbol=True
                    )
                    dim_chunk_val = self.dim_chunk(available_RAM=8)
                    evaluator = SparseEvaluation(self.zonotope_espilon_sparse_tensor, chunk_size=dim_chunk_val, function= lambda x:x , mask_coef=self.mask_epsilon)
                    self.zonotope_espilon_sparse_tensor, self.sum_abs = evaluator.evaluate_all_chunks(num_workers=num_workers)
                    if new_sparse is not None:
                        zono_size = list(self.zonotope_espilon_sparse_tensor.size())
                        new_sparse_size = list(new_sparse.size())
                        print("creation de nouveaux symbols",new_sparse_size[0])
                        new_size  =[zono_size[0]+new_sparse_size[0],*zono_size[1:]]
                        indices = torch.cat([self.zonotope_espilon_sparse_tensor.indices(), new_sparse.indices()], dim=1)
                        values = torch.cat([self.zonotope_espilon_sparse_tensor.values(), new_sparse.values()])
                        self.zonotope_espilon_sparse_tensor = torch.sparse_coo_tensor(indices, values, size = new_size).coalesce()
                    print("nombre de symboles = ",self.zonotope_espilon_sparse_tensor.size())
                    print(self.zonotope_espilon_sparse_tensor)
                
                results[name] = {
                    'center': self.input,
                    'trash_layer': self.trash,
                    'mask_epsilon': self.mask_epsilon,
                    'new_sparse':_
                }
        return results




class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        
 
        self.fc1 = nn.Linear(in_features=524288, out_features=128)  
        self.fc2 = nn.Linear(in_features=128, out_features=10)  
        self.relu4 = nn.ReLU()

    def forward(self, x):

        x = self.relu1(self.conv1(x))

        x = self.relu2(self.conv2(x))
 

  
        x = x.view(x.size(0), -1)
    
        x = self.relu3(self.fc1(x))
      
        x = self.relu4(self.fc2(x))
        return x

model = SimpleCNN()
input_dim = (3,128, 128
             )

unstacked = UnStackNetwork(model, input_dim)
print(*unstacked.output)


test_input = torch.randn(1, *input_dim)
_,zonotope_espilon_sparse_tensor = ZonoSparseGeneration(test_input,0.001).total_zono()
print(zonotope_espilon_sparse_tensor)
ray.init()
layer_evaluator = LayerEvaluator(unstacked.output, test_input)

results = layer_evaluator.evaluate_layers(zonotope_espilon_sparse_tensor)

for layer_name, result in results.items():
    print(f"Layer: {layer_name}")
    print(f"Center: {result['center']}")
    print(f"Trash Layer: {result['trash_layer']}")
    print(f"Mask Epsilon: {result['mask_epsilon']}")
    print(f"New Sparse: {result['new_sparse']}")
