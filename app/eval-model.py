import torch
import torch.nn as nn
import torchvision.models as models
import copy
import ray

import sys
sys.path.append('app/src')
sys.path.append('./src')


from sparse_evaluation import SparseEvaluation  
from zono_sparse_gen import ZonoSparseGeneration
from unstack_network import UnStackNetwork
from abstract_relu import AbstractReLU


class LayerEvaluator:
    def __init__(self, output, test_input):
        self.output = output
        self.test_input = test_input
        

    def evaluate_layers(self, zono_from_test, num_workers=9):
        results = {}
        mask_epsilon = torch.ones_like(self.test_input)
        trash = torch.zeros_like(self.test_input)
        for name, details in self.output.items():
            print(name)
            print(details)
            if 'original' in details:
                layer = details['original']
                epsilon_layer = details[f'epsilon_{name}']
                dim_chunk_val = dim_chunk(layer)
                
                evaluator = SparseEvaluation(zono_from_test, chunk_size=dim_chunk_val, function=epsilon_layer, mask_coef=mask_epsilon)
                zono_from_test, sum_abs = evaluator.evaluate_all_chunks(num_workers=num_workers)
                start_index = zono_from_test.size(0)
                
                trash_layer = details[f'noise_{name}']
                trash  = trash_layer(trash)
                self.test_input = layer(self.test_input)

                # Determine the activation function's abstract class
            activation_name = details.get('activation', None)
            if activation_name:
                class_name = f"Abstract{activation_name}"
                print(f'class_name= {class_name}')
                AbstractClass = globals().get(class_name)
                if AbstractClass:
                    abstract_instance = AbstractClass()
                    self.test_input, trash_layer, mask_epsilon, new_sparse = abstract_instance.abstract_relu(
                        self.test_input, sum_abs, trash, start_index=start_index, add_symbol=True
                    )
                    if new_sparse :
                        zono_size = list(zono_from_test.size())
                    
                        new_sparse_size = list(new_sparse.size())
                        print("creation de nouveaux symbols",new_sparse_size[0])
                    
                        new_size  =[zono_size[0]+new_sparse_size[0],*zono_size[1:]]
                    

                        indices = torch.cat([zono_from_test.indices(), new_sparse.indices()], dim=1)
                        values = torch.cat([zono_from_test.values(), new_sparse.values()])
                        
                        
                        zono_from_test = torch.sparse_coo_tensor(indices, values, size = new_size).coalesce()
                    print("nombre de symboles = ",zono_from_test.size())
                
                results[name] = {
                    'center': self.test_input,
                    'trash_layer': trash,
                    'mask_epsilon': mask_epsilon,
                    'new_sparse':_
                }
        return results

def dim_chunk(layer):

    return 10



model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)

input_dim = (3, 42, 42)

unstacked = UnStackNetwork(model, input_dim)


test_input = torch.randn(1, *input_dim)
_,zono_from_test = ZonoSparseGeneration(test_input,0.00000000001).total_zono()
print(zono_from_test)
ray.init()
layer_evaluator = LayerEvaluator(unstacked.output, test_input)

results = layer_evaluator.evaluate_layers(zono_from_test)

for layer_name, result in results.items():
    print(f"Layer: {layer_name}")
    print(f"Center: {result['center']}")
    print(f"Trash Layer: {result['trash_layer']}")
    print(f"Mask Epsilon: {result['mask_epsilon']}")
    print(f"New Sparse: {result['new_sparse']}")
