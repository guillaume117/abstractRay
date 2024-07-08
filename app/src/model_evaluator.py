import torch
import torch.nn as nn


import sys

sys.path.append('app/src')
sys.path.append('./src')
from util import sparse_tensor_stats, get_largest_tensor_size,sparse_dense_broadcast_mult


from sparse_evaluation_4 import SparseEvaluation  
from zono_sparse_gen import ZonoSparseGeneration

from abstract_relu import AbstractReLU
from sparse_addition_2 import SparseAddition



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
        with torch.no_grad():
            layer = self.details['original']
            print(layer)
            self.input = layer(self.input)

    @staticmethod   
    def static_process_center_layer(input, function):
        with torch.no_grad():
            return function(input)

    def process_trash_layer(self):
        with torch.no_grad():
            trash_layer = self.details[f'noise_{self.name}']
            self.trash = trash_layer(self.trash)
    
    @staticmethod
    def static_process_trash_layer(input,function):
        with torch.no_grad():
            function=function.to('cpu')
            input = input.to('cpu')
            return function(input)


    def process_max_pool2D(self,maxpool,numworkers= None):
        with torch.no_grad():
            if numworkers is None: 
                num_workers = self.num_workers

            dim_x = self.input.size(1)
            kernel_size = maxpool.kernel_size
            
            assert kernel_size==2 or kernel_size==[2,2] ,f"Maxpool2D kernel size {kernel_size}. A kernel size different of 2 is not supported"
            
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

            mask_epsilon = self.mask_epsilon

            E1, S1,c1, t1 = ModelEvaluator.static_process_linear_layer(self.input,
                                                                self.zonotope_espilon_sparse_tensor,
                                                                self.trash,
                                                                conv_0,
                                                                mask_epsilon ,
                                                                conv_0,
                                                                conv_0,
                                                                self.num_workers,
                                                                self.available_RAM,
                                                                self.device)
        
            
            len_E1 = E1.size(0)
            c1, t1, m1= AbstractReLU.abstract_relu(
                            c1, S1,t1, start_index=len_E1, add_symbol=True
                        )
            
            E1,S1,c1, t1 = ModelEvaluator.static_process_linear_layer(c1, E1 ,t1 ,ident, m1 , ident,ident,self.num_workers, self.available_RAM, self.device)
            #print("k"*100)
            #E3=sparse_dense_broadcast_mult(E1,m1)
            #print("k"*100)
            

            E2, S2, c2, t2 = ModelEvaluator.static_process_linear_layer(self.input,
                                                                self.zonotope_espilon_sparse_tensor,
                                                                self.trash,
                                                                conv_1,
                                                                mask_epsilon,
                                                                conv_1,
                                                                conv_1,
                                                                self.num_workers,
                                                                self.available_RAM,
                                                                self.device)
            
            len_E2 = E2.size(0)
            
            #E1,S1 = SparseAddition(E1, E2, chunk_size=chunk_size, device=self.device).addition(num_workers=self.num_workers)
            dim1 = get_largest_tensor_size(E1,E2)
            print(f'dim1 = {dim1}')
            E1 = torch.sparse_coo_tensor(E1.indices(), E1.values(),size=dim1)
            E2 = torch.sparse_coo_tensor(E2.indices(), E2.values(),size=dim1)
            E1 = (E1 + E2).coalesce()
            S1 = torch.sum(torch.abs(E1),dim=0).unsqueeze(0).to_dense()
            c1 =c1+c2
            t1 =torch.abs(t1)+torch.abs(t2)
            E2, S2, c2, t2 = ModelEvaluator.static_process_linear_layer(self.input,
                                                        self.zonotope_espilon_sparse_tensor,
                                                        self.trash,
                                                        conv_2,
                                                        mask_epsilon ,
                                                        conv_2,
                                                        conv_2,
                                                        self.num_workers,
                                                        self.available_RAM,
                                                        self.device)
            chunk_size = ModelEvaluator.static_dim_chunk(c2,self.available_RAM)
            #E2,S2 = SparseAddition(E2, E1, chunk_size=chunk_size, device=self.device).substraction(num_workers=self.num_workers)
            dim1 = get_largest_tensor_size(E1,E2)
            print(f'dim1 = {dim1}')
            E2 = torch.sparse_coo_tensor(E2.indices(), E2.values(),size=dim1).coalesce()
            E1 = torch.sparse_coo_tensor(E1.indices(), E1.values(),size=dim1).coalesce()
            E2 = (E2 - E1).coalesce()
            S2 = torch.sum(torch.abs(E2),dim=0).unsqueeze(0).to_dense()
            c2 =c2-c1
            t2 =torch.abs(t1)+torch.abs(t2)

            len_E2 = E2.size(0)
            c3, t3, m3= AbstractReLU.abstract_relu(
                            c2, S2,t2, start_index=len_E2, add_symbol=True
                        )
            E3,S3,c3,t3 = ModelEvaluator.static_process_linear_layer(c3, E2 ,t3 ,ident, m3 , ident,ident,self.num_workers, self.available_RAM, self.device)
          
            #E3,S3 = SparseAddition(E3, E1, chunk_size=chunk_size, device=self.device).addition(num_workers=self.num_workers)
            dim1 = get_largest_tensor_size(E1,E3)
            print(f'dim1 = {dim1}')
            E3 = torch.sparse_coo_tensor(E3.indices(), E3.values(),size=dim1).coalesce()
            E1 = torch.sparse_coo_tensor(E1.indices(), E1.values(),size=dim1).coalesce()
            E3 = (E3 + E1).coalesce()
            S3 = torch.sum(torch.abs(E3),dim=0).unsqueeze(0).to_dense()  
            c3 =c3+c1
            t3 =torch.abs(t3)+torch.abs(t1)

            E2, S2,c2, t2 = ModelEvaluator.static_process_linear_layer(self.input,
                                                                self.zonotope_espilon_sparse_tensor,
                                                                self.trash,
                                                                conv_3,
                                                                mask_epsilon ,
                                                                conv_3,
                                                                conv_3,
                                                                self.num_workers,
                                                                self.available_RAM,
                                                                self.device)
           
            #E2,S2 = SparseAddition(E2, E3, chunk_size=chunk_size, device=self.device).substraction(num_workers=self.num_workers)
            dim1 = get_largest_tensor_size(E2,E3)
            print(f'dim1 = {dim1}')
            E3 = torch.sparse_coo_tensor(E3.indices(), E3.values(),size=dim1).coalesce()
            E2 = torch.sparse_coo_tensor(E2.indices(), E2.values(),size=dim1).coalesce()
            E2 = (E2 - E3).coalesce()
            S2 = torch.sum(torch.abs(E2),dim=0).unsqueeze(0).to_dense()             
            c2 =c2-c3
            t2 =torch.abs(t3)+torch.abs(t2)
            print(torch.sum(t2)*10)
            len_E2 = E2.size(0)
            c4, t4, m4= AbstractReLU.abstract_relu(
                            c2, S2,t2, start_index=len_E2, add_symbol=True
                        )
        
            
            E4,S4,c4,t4 = ModelEvaluator.static_process_linear_layer(c4, E2 ,t4 ,ident, m4 , ident,ident,self.num_workers, self.available_RAM, self.device)

            
          
            dim1 = get_largest_tensor_size(E1,E3)
            E3 = torch.sparse_coo_tensor(E3.indices(), E3.values(),size=dim1).coalesce()
            E4 = torch.sparse_coo_tensor(E4.indices(), E4.values(),size=dim1).coalesce()
            E4 = (E3 + E4).coalesce()
            S4 = torch.sum(torch.abs(E4),dim=0).unsqueeze(0).to_dense()             
            #E4,S4 = SparseAddition(E4, E3, chunk_size=chunk_size, device=self.device).addition(num_workers=self.num_workers)

            print(E4.size())
            del E1, E2, E3  
            return E4,S4, c4 +c3, torch.abs(t4)+torch.abs(t3)



    @staticmethod
    def static_process_linear_layer(input, zono,trash,function_tot, mask_epsilon , function_abs,function_trash,num_workers, available_RAM, device):
        with torch.no_grad():
            
            dim_chunk_val_input = ModelEvaluator.static_dim_chunk(input, available_RAM)
            center  = ModelEvaluator.static_process_center_layer(input,function_tot)
            sum_abs =torch.zeros_like(center)
            dim_chunk_val_output = ModelEvaluator.static_dim_chunk(center,available_RAM)

            dim_chunk_val = min(dim_chunk_val_input, dim_chunk_val_output)
        
        
            evaluator = SparseEvaluation(zono,
                                        chunk_size=dim_chunk_val, 
                                        function=function_abs, 
                                        mask_coef=mask_epsilon, 
                                        device= device)
            
            zono_out, sum = evaluator.evaluate_all_chunks(num_workers=num_workers)
            len_zono = zono_out.size(0)
            mask_epsilon = torch.ones_like(mask_epsilon)
          
            _, new_sparse = ZonoSparseGeneration(trash,from_trash=True,start_index=len_zono).total_zono()
            
            
            if new_sparse is not None:
                print(f'new_sparse size {new_sparse.size()}')
                evaluator_new_noise = SparseEvaluation(new_sparse,
                                                    chunk_size = dim_chunk_val,
                                                    function=function_abs, 
                                                    mask_coef = mask_epsilon,
                                                    eval_start_index=len_zono,
                                                    device =device)
                new_sparse, sum = evaluator_new_noise.evaluate_all_chunks(num_workers=num_workers)
                sum_abs +=sum
                zono_size = list(zono_out.size())
                new_sparse_size = list(new_sparse.size())
            
                new_size  =[zono_size[0]+new_sparse._nnz(),*zono_size[1:]]
    
                
                zono_out = torch.sparse_coo_tensor(zono_out.indices(), zono_out.values(), size = new_sparse.size()).coalesce()
                zono_out += new_sparse
            sum_abs = torch.sum(torch.abs(zono_out),dim=0).unsqueeze(0).to_dense()
              
                
            new_sparse = None



            trash = ModelEvaluator.static_process_trash_layer(trash,function_trash)
            
            trash = torch.zeros_like(trash)
            return zono_out, sum_abs, center, trash



    def process_linear_layer(self, num_workers=None, epsilon_layer = None):
        with torch.no_grad():
            if num_workers is None:
                num_workers =self.num_workers
            
            
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
            
        
            
            if new_sparse is not None:
                print(f'new_sparse size {new_sparse.size()}')
                evaluator_new_noise = SparseEvaluation(new_sparse,
                                                    chunk_size = dim_chunk_val,
                                                    function=epsilon_layer, 
                                                    mask_coef = self.mask_epsilon,
                                                    eval_start_index=self.len_zono,
                                                    device =self.device)
                new_sparse, _ = evaluator_new_noise.evaluate_all_chunks(num_workers=self.num_workers)
               
                zono_size = list(self.zonotope_espilon_sparse_tensor.size())
               
                
                #indices = torch.cat([self.zonotope_espilon_sparse_tensor.indices(), new_sparse.indices()], dim=1)
                self.zonotope_espilon_sparse_tensor= torch.sparse_coo_tensor(self.zonotope_espilon_sparse_tensor.indices(),self.zonotope_espilon_sparse_tensor.values(), new_sparse.size()).coalesce()
                self.zonotope_espilon_sparse_tensor += new_sparse
            self.sum_abs = torch.sum(torch.abs(self.zonotope_espilon_sparse_tensor),dim=0).unsqueeze(0).to_dense()
                #torch.sparse_coo_tensor(indices, values, size = new_sparse.size()).coalesce()
            new_sparse = None



            self.process_trash_layer()
            
            self.trash = torch.zeros_like(self.trash)


    

    
    def evaluate_model(self, zonotope_espilon_sparse_tensor, num_workers=None):
        with torch.no_grad():
        
            if num_workers is None:
                num_workers = self.num_workers
                
    
            self.zonotope_espilon_sparse_tensor=zonotope_espilon_sparse_tensor
            results = {}
            self.mask_epsilon = torch.ones_like(self.input)
            self.trash = torch.zeros_like(self.input)



            for name, details in self.output.items():
                print("#"*100)
                print(f'layer name {name}')
                print("-"*100)
                print(f'layer details {details}')
                print("#"*100)
                self.name = name
                self.details = details


                non_zero_percentage, memory_gain_percentage = sparse_tensor_stats(self.zonotope_espilon_sparse_tensor)
                print(f"Pourcentage de valeurs non nulles: {non_zero_percentage:.2f}%")
                print(f"Gain en pourcentage de mémoire: {memory_gain_percentage:.2f}%")



                if 'original' in details:

                    self.process_linear_layer()
                    self.mask_epsilon = torch.ones_like(self.input)

                    
                    
                
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