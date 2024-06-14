import sys
sys.path.append('app/src')
sys.path.append('./src')
import torch 
import torch.nn as nn 
from typing import List, Union, Tuple, Callable
from torch.sparse import FloatTensor


def list_of_shape(tensor):
    tensor= torch.tensor(tensor)
    return [shape for shape in tensor.shape]

class ZonoSparseGeneration :

    "This class aims to genereate a sparse representation of abstract domain"
    def __init__(self, input: FloatTensor, noise_intensity:Union[float,torch.tensor], noise_type : str ='additive', indices = None ,from_trash = False,start_index = None):
        self.input = input
        self.noise_intensity = torch.tensor(noise_intensity)
        self.noise_type = noise_type
        self.input_shape = list_of_shape(input)
        self.indices = indices
        self.from_trash = from_trash
        self.start_index = start_index
        

        
     

    def total_zono(self):
        if not self.from_trash: 


            dim_input = torch.tensor(self.input_shape).numel()
        
            if dim_input ==1 :
                global_storage = {
                    'indices': [],
                    'values': []
                }
                if self.indices is None: 
                    num_elements = self.input_shape[0]
                    self.indices = torch.arange(1,num_elements,1)
                else:
                    self.indices = torch.tensor(self.indices)
                    num_elements = self.indices.numel()
                if len(self.noise_intensity.flatten())==1:
                    self.noise_intensity = self.noise_intensity*torch.ones_like(self.indices)
                else : 
                    assert (self.noise_intensity.size()==self.indices.size()); 'the len of noise intensity must be one or equal to indice shape '

                for i in range(num_elements):
                    global_storage['indices'].append([self.indices[i],self.indices[i]])
                    global_storage['values'].append(self.noise_intensity[i])
                indice_tensor   = torch.tensor(global_storage['indices'], dtype = torch.int32).t()
                
                values_tensor   = torch.tensor(global_storage['values'] , dtype = torch.float32)
                sparse_zonotope = torch.sparse_coo_tensor(indice_tensor,values_tensor,size = (self.input_shape[0],self.input_shape[0]))
                
                return self.input, sparse_zonotope.to_dense()             


                
            if dim_input ==2 :
                self.input = self.input.unsqueeze(0)
                self.input_shape = list_of_shape(self.input)
            if dim_input == 4 : 
                self.input = self.input.squeeze(0)
                print("WARNING : Trying to generate abstract Sparse tensor from a batch, only the first element will be used")
                self.input_shape = list_of_shape(self.input)
            if self.indices is None: 
                assert (len(self.noise_intensity.flatten())==1);'Shape of noise and indices do not match'

                num_elements = self.input_shape[0]
                self.indices = torch.arange(1,num_elements,1)

                global_storage = {
                        'indices': [],
                        'values': []
                    }
                num_elements = self.input_shape[0]*self.input_shape[1]*self.input_shape[2]
                for i in range(num_elements):
                    dim_3 = i // (self.input_shape[1]*self.input_shape[2])
                    rem = i % (self.input_shape[1]*self.input_shape[2])
                    dim_1 = rem // self.input_shape[1]
                    dim_2 = rem % self.input_shape[2]
                    global_storage['indices'].append([i,dim_3,dim_1,dim_2])
                    global_storage['values'].append(self.noise_intensity)
                    

                indice_tensor = torch.tensor(global_storage['indices'],dtype = torch.int32).t()
                values_tensor = torch.tensor(global_storage['values'], dtype = torch.float32)

                sparse_zonotope = torch.sparse_coo_tensor(indice_tensor,values_tensor, size=(num_elements,self.input_shape[0],self.input_shape[1],self.input_shape[2]))


            else :
                self.indices = torch.tensor(self.indices)

    
                assert len(self.indices)==len(self.noise_intensity); 'Lenght of Noise_intensity and indice mismatch'
                global_storage = {
                        'indices': [],
                        'values': []
                    }       
                num_elements =len(self.indices)
                for i in range(len(self.indices)):
                    if len(self.indices[i])==2:
                        global_storage['indices'].append(torch.cat((torch.tensor([i,0]),self.indices[i])).tolist())
                        
                    else :
                        global_storage['indices'].append(torch.cat((torch.tensor([i]),self.indices[i])).tolist())
                    global_storage['values'].append(self.noise_intensity[i])
                indice_tensor = torch.tensor(global_storage['indices'],dtype = torch.int32).t()
                values_tensor = torch.tensor(global_storage['values'],dtype=torch.float32)
                print(indice_tensor)
                print(values_tensor)




                sparse_zonotope = torch.sparse_coo_tensor(indice_tensor,values_tensor, size=(num_elements,self.input_shape[0],self.input_shape[1],self.input_shape[2]))
                

            return self.input, sparse_zonotope
                
        if self.from_trash:
            if not self.start_index:
                print('Warning, start_index is 0, should start at the depth of abstract domain')
                self.start_index =0
            
            global_storage = {
                        'indices': [],
                        'values': []
                    } 
            indices = torch.nonzero(self.input)
            print(indices)
            for i,indice in enumerate(indices):
           
                sparse_indice = torch.cat((torch.tensor([i+self.start_index]),indice)).tolist()
                global_storage['indices'].append(sparse_indice)
                global_storage['values'].append(self.input[tuple(indice.tolist())])
            indice_tensor = torch.tensor(global_storage['indices'],dtype = torch.int32).t()
            values_tensor = torch.tensor(global_storage['values'],dtype=torch.float32)
            dim = tuple(torch.cat((torch.tensor([len(indices)]),torch.tensor(list_of_shape(self.input)))))

            
            sparse_zonotope = torch.sparse_coo_tensor(indice_tensor,values_tensor, size =dim)

            return self.input, sparse_zonotope


        


def main():
    test = torch.randn(1,3,3)
    center, zono = ZonoSparseGeneration(test, noise_intensity=[1,1], indices = torch.tensor([[3,3],[4,4]]),from_trash=True,start_index=1321).total_zono()
    print(center)
    print(zono)
  
 


if __name__ == "__main__":
    main()