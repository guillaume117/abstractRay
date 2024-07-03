import torch
import torch.nn as nn
import torchvision.models as models
import copy
import ray
from PIL import Image
import sys
from torchvision import transforms
sys.path.append('app/src')
sys.path.append('./src')



from sparse_evaluation_4 import SparseEvaluation  
from zono_sparse_gen import ZonoSparseGeneration
from unstack_network import UnStackNetwork
from abstract_relu import AbstractReLU
from sparse_addition_2 import SparseAddition
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

    def __init__(self, unstacked_model, input,num_workers = 0,available_RAM = 8, device =torch.device('cpu')):
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
    

    @staticmethod
    def static_dim_chunk(input,available_RAM):
        dense_memory_footprint = torch.prod(torch.tensor(input.shape)) *4/1e9
        return int(max(1,available_RAM//(3*dense_memory_footprint)))


    
    def process_center_layer(self):

        layer = self.details['original']
        print(layer)
        self.input = layer(self.input)

    @staticmethod   
    def static_process_center_layer(input, function):
        return function(input)

    def process_trash_layer(self):

        trash_layer = self.details[f'noise_{self.name}']
        self.trash = trash_layer(self.trash)
    
    @staticmethod
    def static_process_trash_layer(input,function):
        function=function.to('cpu')
        input = input.to('cpu')
        return function(input)


    def process_max_pool2D(self,maxpool,numworkers= None):
        if numworkers is None: 
            num_workers = self.num_workers

        dim_x = self.input.size(1)
        kernel_size = maxpool.kernel_size
        
        assert kernel_size==2,f"Maxpool2D kernel size {kernel_size}. A kernel size different of 2 is not supported"
        
        stride = maxpool.stride
        padding = maxpool.padding
        kernel_size = 2





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

        ident = nn.Identity()



        E1, S1,c1, t1 = ModelEvaluator.static_process_linear_layer(self.input,
                                                            self.zonotope_espilon_sparse_tensor,
                                                            self.trash,
                                                            conv_0,
                                                            self.mask_epsilon ,
                                                            conv_0,
                                                            conv_0,
                                                            self.num_workers,
                                                            self.available_RAM,
                                                            self.device)
    
        
        len_E1 = E1.size(0)
        x1, t1, m1= AbstractReLU.abstract_relu(
                        c1, S1,t1, start_index=len_E1, add_symbol=True
                    )
        E1,S1,c1, t1 = ModelEvaluator.static_process_linear_layer(x1, E1 ,t1 ,ident, m1 , ident,ident,self.num_workers, self.available_RAM, self.device)
        

        E2, S2, c2, t2 = ModelEvaluator.static_process_linear_layer(self.input,
                                                            self.zonotope_espilon_sparse_tensor,
                                                            self.trash,
                                                            conv_1,
                                                            self.mask_epsilon,
                                                            conv_1,
                                                            conv_1,
                                                            self.num_workers,
                                                            self.available_RAM,
                                                            self.device)
        
        len_E2 = E2.size(0)
        chunk_size = ModelEvaluator.static_dim_chunk(c2,self.available_RAM)
        E1,S1 = SparseAddition(E1, E2, chunk_size=chunk_size, device=self.device).addition(num_workers=self.num_workers)
        c1 =c1+c2
        t1 =torch.abs(t1)+torch.abs(t2)
        E2, S2, c2, t2 = ModelEvaluator.static_process_linear_layer(self.input,
                                                    self.zonotope_espilon_sparse_tensor,
                                                    self.trash,
                                                    conv_2,
                                                    self.mask_epsilon ,
                                                    conv_2,
                                                    conv_2,
                                                    self.num_workers,
                                                    self.available_RAM,
                                                    self.device)
        chunk_size = ModelEvaluator.static_dim_chunk(c2,self.available_RAM)
        E2,S2 = SparseAddition(E2, E1, chunk_size=chunk_size, device=self.device).substraction(num_workers=self.num_workers)
        c2 =c2-c1
        t2 =torch.abs(t1)+torch.abs(t2)

        len_E2 = E2.size(0)
        x3, t3, m3= AbstractReLU.abstract_relu(
                        c2, S2,t2, start_index=len_E2, add_symbol=True
                    )
        E3,S3,c3,t3 = ModelEvaluator.static_process_linear_layer(x3, E2 ,t3 ,ident, m3 , ident,ident,self.num_workers, self.available_RAM, self.device)
        chunk_size = ModelEvaluator.static_dim_chunk(c3,self.available_RAM)
        E3,S3 = SparseAddition(E3, E1, chunk_size=chunk_size, device=self.device).addition(num_workers=self.num_workers)
        c3 =c3+c1
        t3 =torch.abs(t3)+torch.abs(t1)

        E2, S2,c2, t2 = ModelEvaluator.static_process_linear_layer(self.input,
                                                            self.zonotope_espilon_sparse_tensor,
                                                            self.trash,
                                                            conv_3,
                                                            self.mask_epsilon ,
                                                            conv_3,
                                                            conv_3,
                                                            self.num_workers,
                                                            self.available_RAM,
                                                            self.device)
        chunk_size = ModelEvaluator.static_dim_chunk(c2,self.available_RAM)
        E2,S2 = SparseAddition(E2, E3, chunk_size=chunk_size, device=self.device).substraction(num_workers=self.num_workers)
        c2 =c2-c3
        t2 =torch.abs(t3)+torch.abs(t2)
        len_E2 = E2.size(0)
        x4, t4, m4= AbstractReLU.abstract_relu(
                        c2, S2,t2, start_index=len_E2, add_symbol=True
                    )
       
        
        E4,S4,c4,t4 = ModelEvaluator.static_process_linear_layer(x4, E2 ,t4 ,ident, m4 , ident,ident,self.num_workers, self.available_RAM, self.device)

        
        chunk_size = ModelEvaluator.static_dim_chunk(c4,self.available_RAM)
        E4,S4 = SparseAddition(E4, E3, chunk_size=chunk_size, device=self.device).addition(num_workers=self.num_workers)

        print(E4.size())
        return E4,S4, c4 +c3, torch.abs(t4)+torch.abs(t3)



    @staticmethod
    def static_process_linear_layer(input, zono,trash,function_tot, mask_epsilon , function_abs,function_trash,num_workers, available_RAM, device):
        sum_abs =0
        dim_chunk_val_input = ModelEvaluator.static_dim_chunk(input, available_RAM)
        center  = ModelEvaluator.static_process_center_layer(input,function_tot)
        dim_chunk_val_output = ModelEvaluator.static_dim_chunk(center,available_RAM)

        dim_chunk_val = min(dim_chunk_val_input, dim_chunk_val_output)
       
    
        evaluator = SparseEvaluation(zono,
                                     chunk_size=dim_chunk_val, 
                                     function=function_abs, 
                                     mask_coef=mask_epsilon, 
                                     device= device)
        
        zono, sum = evaluator.evaluate_all_chunks(num_workers=num_workers)
        len_zono = zono.size(0)
        mask_epsilon = torch.ones_like(mask_epsilon)

        _, new_sparse = ZonoSparseGeneration(trash,from_trash=True,start_index=len_zono).total_zono()
        print(new_sparse)
        
        if new_sparse is not None:
            evaluator_new_noise = SparseEvaluation(new_sparse,
                                                   chunk_size = dim_chunk_val,
                                                   function=function_abs, 
                                                   mask_coef = mask_epsilon,
                                                   eval_start_index=len_zono,
                                                   device =device)
            new_sparse, sum = evaluator_new_noise.evaluate_all_chunks(num_workers=num_workers)
            sum_abs +=sum
            zono_size = list(zono.size())
            new_sparse_size = list(new_sparse.size())
         
            new_size  =[zono_size[0]+new_sparse_size[0],*zono_size[1:]]
   
            indices = torch.cat([zono.indices(), new_sparse.indices()], dim=1)
            values = torch.cat([zono.values(), new_sparse.values()])
            zono = torch.sparse_coo_tensor(indices, values, size = new_size).coalesce()
        new_sparse = None



        trash = ModelEvaluator.static_process_trash_layer(trash,function_trash)
        
        trash = torch.zeros_like(trash)
        return zono, sum_abs,center, trash



    def process_linear_layer(self, num_workers=None, epsilon_layer = None):

        if num_workers is None:
            num_workers =self.num_workers
        
        
        dim_chunk_val_input = self.dim_chunk()
        self.process_center_layer()
        dim_chunk_val_output = self.dim_chunk()

        dim_chunk_val = min(dim_chunk_val_input, dim_chunk_val_output)
        if epsilon_layer is None : 

            epsilon_layer = self.details[f'epsilon_{self.name}']
            print("8"*100)
            print(epsilon_layer)
    
        evaluator = SparseEvaluation(self.zonotope_espilon_sparse_tensor, 
                                     chunk_size=dim_chunk_val, 
                                     function=epsilon_layer, 
                                     mask_coef=self.mask_epsilon, 
                                     device= self.device)
        
        self.zonotope_espilon_sparse_tensor, self.sum_abs = evaluator.evaluate_all_chunks(num_workers=num_workers)
        self.len_zono = self.zonotope_espilon_sparse_tensor.size(0)
        self.mask_epsilon = torch.ones_like(self.mask_epsilon)

        _, new_sparse = ZonoSparseGeneration(self.trash,from_trash=True,start_index=self.len_zono).total_zono()
      
        
        if new_sparse is not None:
            evaluator_new_noise = SparseEvaluation(new_sparse,
                                                   chunk_size = dim_chunk_val,
                                                   function=epsilon_layer, 
                                                   mask_coef = self.mask_epsilon,
                                                   eval_start_index=self.len_zono,
                                                   device =self.device)
            new_sparse, sum = evaluator_new_noise.evaluate_all_chunks(num_workers=self.num_workers)
            self.sum_abs +=sum
            zono_size = list(self.zonotope_espilon_sparse_tensor.size())
            new_sparse_size = list(new_sparse.size())
           
            new_size  =[zono_size[0]+new_sparse_size[0],*zono_size[1:]]
            
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
                self.mask_epsilon = torch.ones_like(self.input)

                print('name passed = ',self.name)
                
               
            activation_name = self.details.get('activation', None)


            if activation_name:
                if activation_name == 'MaxPool2d':
                    self.zonotope_espilon_sparse_tensor,self.sum_abs,self.input,self.trash = self.process_max_pool2D(self.details['activation_function'])
                    self.mask_epsilon = torch.ones_like(self.input)
                
                else : 
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
        
        self.maxpool2D = nn.MaxPool2d(kernel_size=2,stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(8,8))
        
 
        self.fc1 = nn.Linear(in_features=2048, out_features=512)  
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=512, out_features=10)  
        self.relu4 = nn.ReLU()

        #self.fc3 = nn.Linear(in_features=128, out_features=10)  
        #self.relu5 = nn.ReLU()

    def forward(self, x):

        x = self.relu1(self.conv1(x))
        

        x = self.relu2(self.conv2(x))

       

        x=self.maxpool2D(x)
        
       
        x = self.avgpool(x)
  
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
      
        x = self.relu4(self.fc2(x))
      #  x = self.relu5(self.fc3(x))
        return x
    
model = SimpleCNN()
input_dim =(3,112,112)
"""
import onnx
from onnx2torch import convert
path = './vgg16-7.onnx'
onnx_model = onnx.load(path)
pytorch_model = convert(onnx_model)
model = pytorch_model

"""



input_dim = (3,112,112
             )

image_path = "app/output_image.jpeg"
image = Image.open(image_path)


transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
])

test_input = transform(image).unsqueeze(0) 

print(f"True:{model(test_input)}")


unstacked = UnStackNetwork(model, input_dim)
print("*"*100)
print("unstacked output ",*unstacked.output)
print("*"*100)

_,zonotope_espilon_sparse_tensor = ZonoSparseGeneration(test_input,0.001).total_zono()
print(zonotope_espilon_sparse_tensor)
ray.init()
model_evaluator = ModelEvaluator(unstacked.output, test_input,num_workers=0, available_RAM=10,device=torch.device('cpu'))

result = model_evaluator.evaluate_model(zonotope_espilon_sparse_tensor)

print(f"True:{model(test_input)}")
print(f"Center: {result['center']}")
print(f"Min: {result['min']}")
print(f"Max: {result['max']}")
print(f"Relevance: {result['relevance']}")
