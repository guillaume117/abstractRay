import torch
import torch.nn.functional as F
from torch.sparse import FloatTensor
from typing import Callable
import torch.nn as nn
import ray
from tqdm import tqdm
import sys 
sys.path.append('cpuconv2D')
sys.path.append('src/cpuconv2D')
sys.path.append('app/src/cpuconv2D')
sys.path.append('./app')
sys.path.append('./app/backend')
sys.path.append('./app/backend/src')
sys.path.append('./app/backend/src/cpuconv2D')
#import sparse_conv2d
import os
os.environ["RAY_NUM_CPUS"] = str(os.cpu_count())



dtyped =torch.long

@ray.remote(num_gpus=1)
class SparseWorker:
    def __init__(self, x_chunk, chunk_size, mask_coef, function, dense_shape, worker_start_index, device,verbose = False):
        with torch.no_grad():
            self.x_chunk = x_chunk.coalesce().to(device)
            self.chunk_size = chunk_size
            self.mask_coef = mask_coef.to(device)
            if isinstance(function, nn.Module):
                function = function.to(device)
            self.function = function
            self.dense_shape = dense_shape
            self.worker_start_index = worker_start_index
            self.device = device
            self.verbose = verbose

    def evaluate_chunks(self):
        with torch.no_grad():
            indices = self.x_chunk.indices().t()
            values = self.x_chunk.values()

            global_storage = {
                'indices': [],
                'values': []
            }
     

            num_chunks = (self.x_chunk.size(0) + self.chunk_size - 1) // self.chunk_size

            for i in range(num_chunks):
                
                chunk_start = i * self.chunk_size
                chunk_end = min((i + 1) * self.chunk_size, self.x_chunk.size(0))
                mask = (indices[:, 0] >= chunk_start) & (indices[:, 0] < chunk_end)

                chunk_indices = indices[mask]
                chunk_indices[:, 0] -= chunk_start

                chunk_values = values[mask]
                chunk_size = chunk_end - chunk_start

                chunk_sparse_tensor = torch.sparse_coo_tensor(
                    chunk_indices.t(), chunk_values,
                    torch.Size([chunk_size] + self.dense_shape[1:])
                ).coalesce()

                chunk_dense_tensor = chunk_sparse_tensor.to_dense().to(self.device)
                func_output = self.function(self.mask_coef * chunk_dense_tensor)

     

                func_output_sparse = func_output.to_sparse().to('cpu').coalesce()
                add_indices = func_output_sparse.indices().to(dtyped) + torch.tensor(
                    [[chunk_start + self.worker_start_index]] + [[0]] * (func_output_sparse.indices().size(0) - 1), dtype=dtyped, device=torch.device('cpu')
                )

                global_storage['indices'].append(add_indices.cpu())
                global_storage['values'].append(func_output_sparse.values().cpu())

                del chunk_dense_tensor, chunk_sparse_tensor, func_output, func_output_sparse, add_indices
                torch.cuda.empty_cache()
                progress = (i + 1) / num_chunks
                if self.verbose : 
                    if i % 10 == 0:
                        print(f"Worker {self.worker_start_index // self.dense_shape[0]} progress: {progress:.2%}")

            if global_storage['indices']:
                global_indices = torch.cat(global_storage['indices'], dim=1)
                global_values = torch.cat(global_storage['values'], dim=0)
            else:
                global_indices = torch.empty((2, 0), dtype=dtyped)
                global_values = torch.empty((0,), dtype=torch.float32)

        return global_indices, global_values

class SparseEvaluation:
    def __init__(self, x: FloatTensor, chunk_size: int, mask_coef: FloatTensor = None, function: Callable = None, eval_start_index=0, device=torch.device('cpu'),verbose = False):

        
        
        with torch.no_grad():   
            self.x = x
            self.chunk_size = chunk_size
            self.dense_shape = list(x.size())
            self.device = device
            self.eval_start_index = eval_start_index
            self.conv2d_type = False
            self.verbose = verbose
            
            if self.device == torch.device('cuda'):
                self.num_gpus = 1
                self.num_cpus = os.cpu_count()
            else:
                self.num_gpus = 0
                self.num_cpus = os.cpu_count()



            if isinstance(function, nn.Conv2d):
                self.conv2d_type = False
            if self.conv2d_type: #function = nn.Conv2d()
                weights_tensor = function.weight.data
                bias = []
                in_chanels = function.in_channels
                out_chanels = function.out_channels
                kernel_size = function.kernel_size[0]
                stride = function.stride[0]
                padding = function.padding[0]
                groups = function.groups

                weights = weights_tensor.numpy().flatten().tolist()
              
           

                #self.conv = sparse_conv2d.SparseConv2D(in_chanels, out_chanels, kernel_size, stride, padding,groups, weights,bias)

                
            if function is None:
                self.function = nn.Indentity()
            else:
                self.function = function
            func_copy=self.function.to('cpu')

            x0 = torch.zeros(1, *self.dense_shape[1:])
            if mask_coef is None:
                self.mask_coef = torch.ones_like(x0).to(self.device)
            else:
                self.mask_coef = mask_coef.to(self.device)

            self.num_chunks = (self.dense_shape[0] + self.chunk_size - 1) // self.chunk_size
            self.output_size = list(func_copy(x0).shape)
            self.output_size[0] = self.dense_shape[0]
    

    


    def evaluate_all_chunks(self, num_workers):
        
        with torch.no_grad():
            if num_workers == 0:
                return self.evaluate_chunks_directly()
            
            indices = self.x.indices()
            values = self.x.values()
            indices = indices.t()

            chunk_size_per_worker = (self.dense_shape[0] + num_workers - 1) // num_workers
            workers = []
            for i in range(num_workers):
                chunk_start = i * chunk_size_per_worker
                chunk_end = min((i + 1) * chunk_size_per_worker, self.dense_shape[0])
                mask = (indices[:, 0] >= chunk_start) & (indices[:, 0] < chunk_end)

                worker_indices = indices[mask]
                worker_indices[:, 0] -= chunk_start

                worker_values = values[mask]
                worker_size = chunk_end - chunk_start

                if worker_indices.size(0) == 0:
                    continue

                worker_sparse_tensor = torch.sparse_coo_tensor(
                    worker_indices.t(), worker_values,
                    torch.Size([worker_size] + self.dense_shape[1:])
                ).coalesce()
                
                worker = SparseWorker.remote(worker_sparse_tensor, self.chunk_size, self.mask_coef, self.function, self.dense_shape, chunk_start, self.device)
                workers.append(worker.evaluate_chunks.remote())

            results = ray.get(workers)
            global_indices = []
            global_values = []
     

            for add_indices, func_values in results:
                add_indices = add_indices + torch.tensor(
                    [[self.eval_start_index]] + [[0]] * (add_indices.size(0) - 1), dtype=dtyped, device=torch.device('cpu')
                )
                global_indices.append(add_indices)
                global_values.append(func_values)
  

            if global_indices:
                global_indices = torch.cat(global_indices, dim=1)
                global_values = torch.cat(global_values, dim=0)
            else:
                global_indices = torch.empty((2, 0), dtype=dtyped)
                global_values = torch.empty((0,), dtype=torch.float32)

            global_sparse_tensor = torch.sparse_coo_tensor(global_indices, global_values, size=self.output_size).coalesce().to('cpu')

        return global_sparse_tensor

    def evaluate_chunks_directly(self):
        #if self.conv2d_type == True:
        #    return self.evaluate_zono_directly()
        
        with torch.no_grad():
            indices = self.x.indices().t()
            values = self.x.values()
            self.function = self.function.to(self.device)

            global_storage = {
                'indices': [],
                'values': []
            }
        

            num_chunks = (self.x.size(0) + self.chunk_size - 1) // self.chunk_size

            for i in range(num_chunks):
                with torch.no_grad():
                    chunk_start = i * self.chunk_size
                    chunk_end = min((i + 1) * self.chunk_size, self.x.size(0))
                    mask = (indices[:, 0] >= chunk_start) & (indices[:, 0] < chunk_end)

                    chunk_indices = indices[mask]
                    chunk_indices[:, 0] -= chunk_start

                    chunk_values = values[mask]
                    chunk_size = chunk_end - chunk_start

                   

                    chunk_sparse_tensor = torch.sparse_coo_tensor(
                        chunk_indices.t(), chunk_values,
                        torch.Size([chunk_size] + self.dense_shape[1:])
                    ).coalesce()
                    if self.conv2d_type:
                 
                        func_output_sparse = self.conv(chunk_sparse_tensor,self.mask_coef.squeeze(0)).coalesce()
                      
                    else: 

                        chunk_dense_tensor = chunk_sparse_tensor.to_dense().to(self.device)
                        func_output = self.function(self.mask_coef * chunk_dense_tensor)
                        func_output_sparse = func_output.to_sparse().to('cpu').coalesce()
                        del chunk_dense_tensor, func_output

                   
                    add_indices = func_output_sparse.indices().to(dtyped) + torch.tensor(
                        [[chunk_start + self.eval_start_index]] + [[0]] * (func_output_sparse.indices().size(0) - 1), dtype=dtyped, device=torch.device('cpu')
                    )

                    global_storage['indices'].append(add_indices.cpu())
                    global_storage['values'].append(func_output_sparse.values().cpu())

                    del  chunk_sparse_tensor, func_output_sparse, add_indices
                    torch.cuda.empty_cache()

        
            global_indices = torch.cat(global_storage['indices'], dim=1)
            global_values = torch.cat(global_storage['values'], dim=0)
            

            global_sparse_tensor = torch.sparse_coo_tensor(global_indices, global_values, size=self.output_size).coalesce().to('cpu')

        return global_sparse_tensor
    
    def evaluate_zono_directly(self):
        self.x = self.conv(self.x,self.mask_coef.squeeze(0)).coalesce()
        return self.x, torch.sum(torch.abs(self.x))

def test_sparse_evaluation(x):
    

    function = nn.Conv2d(3, 3, 3)
    function.bias.data =torch.zeros_like(function.bias.data)
    eval = SparseEvaluation(x, 100, function=function, device=torch.device('cpu'))
    result = eval.evaluate_chunks_directly()
    sum = torch.sum(result,dim=0)
    with torch.no_grad():
        print(torch.sum(result)- torch.sum(function(x.to_dense())))
        print(f" diff de sum {sum.to_dense()- torch.sum(torch.abs(function(x.to_dense())),dim=0)}")

def test_sparse_evaluation_ray(x):
    

    function = nn.Conv2d(3, 3, 3)
    function.bias.data =torch.zeros_like(function.bias.data)
    eval = SparseEvaluation(x, 100, function=function, device=torch.device('cpu'))
    result = eval.evaluate_all_chunks(num_workers=5)
    sum = torch.sum(result,dim=0)
    with torch.no_grad():
        print(torch.sum(result)- torch.sum(function(x.to_dense())))

        print(f" diff de sum {sum.to_dense()- torch.sum(torch.abs(function(x.to_dense())),dim=0)}")

if __name__ == "__main__":
    x =torch.randn(500, 3, 28, 28).to_sparse(
    )
    test_sparse_evaluation(x)
    test_sparse_evaluation_ray(x)