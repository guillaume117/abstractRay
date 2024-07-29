import torch
import torch.nn.functional as F
from torch.sparse import FloatTensor
from typing import Callable
import torch.nn as nn
import ray
from tqdm import tqdm
import os
import copy


dtyped = torch.long
if os.getenv("RAY_BACKEND") == 'cuda':
    num_gpus = 1
    num_cpus = os.cpu_count()
else:
    num_gpus = 0
    num_cpus = os.cpu_count()

@ray.remote(num_cpus=num_cpus,num_gpus=num_gpus)
class SparseWorker:
    """
    A worker class to evaluate chunks of a sparse tensor using Ray for parallel computation.

    Args:
        x_chunk (torch.sparse.FloatTensor): The chunk of the sparse tensor to evaluate.
        chunk_size (int): The size of each chunk.
        mask_coef (torch.Tensor): The mask coefficient tensor.
        function (Callable): The function to apply to the dense tensor.
        dense_shape (list): The shape of the dense tensor.
        worker_start_index (int): The starting index for the worker.
        device (torch.device): The device to run the evaluation on.
        verbose (bool, optional): Flag to enable verbose output. Defaults to False.
    """
    def __init__(self, x_chunk, chunk_size, mask_coef, function, dense_shape, worker_start_index, device, verbose=False):
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
        """
        Evaluate chunks of the sparse tensor.

        Returns:
            tuple: Indices and values of the evaluated sparse tensor.
        """
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
                if self.verbose:
                    if i % 10 == 0:
                        print(f"Worker {self.worker_start_index // self.dense_shape[0]} progress: {progress:.2%}")

            
            global_indices = torch.cat(global_storage['indices'], dim=1)
            global_values = torch.cat(global_storage['values'], dim=0)
        

        return global_indices, global_values

class SparseEvaluation:
    """
    A class to evaluate sparse tensors in chunks using parallel computation.

    Args:
        x (FloatTensor): The sparse tensor to evaluate.
        chunk_size (int): The size of each chunk.
        mask_coef (FloatTensor, optional): The mask coefficient tensor. Defaults to None.
        function (Callable, optional): The function to apply to the dense tensor. Defaults to None.
        eval_start_index (int, optional): The starting index for evaluation. Defaults to 0.
        device (torch.device, optional): The device to run the evaluation on. Defaults to torch.device('cpu').
        verbose (bool, optional): Flag to enable verbose output. Defaults to False.
    """
    def __init__(self, x: FloatTensor, chunk_size: int, mask_coef: FloatTensor = None, function: Callable = None, eval_start_index=0, device=torch.device('cpu'), verbose=False):
        with torch.no_grad():
            self.x = x
            self.chunk_size = chunk_size
            self.dense_shape = list(x.size())
            self.device = device
            self.eval_start_index = eval_start_index
            self.conv2d_type = False
            self.verbose = verbose



            if isinstance(function, nn.Conv2d):
                self.conv2d_type = False

            if function is None:
                self.function = nn.Identity()
            else:
                self.function = function

            func_copy = self.function.to('cpu')

            x0 = torch.zeros(1, *self.dense_shape[1:])
            if mask_coef is None:
                self.mask_coef = torch.ones_like(x0).to(self.device)
            else:
                self.mask_coef = mask_coef.to(self.device)

            self.num_chunks = (self.dense_shape[0] + self.chunk_size - 1) // self.chunk_size
            self.output_size = list(func_copy(x0).shape)
            self.output_size[0] = self.dense_shape[0]

    def evaluate_all_chunks(self, num_workers):
        """
        Evaluate all chunks using parallel computation.

        Args:
            num_workers (int): Number of workers for parallel computation.

        Returns:
            torch.sparse.FloatTensor: The evaluated sparse tensor.
        """
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

           
            global_indices = torch.cat(global_indices, dim=1)
            global_values = torch.cat(global_values, dim=0)
   

            global_sparse_tensor = torch.sparse_coo_tensor(global_indices, global_values, size=self.output_size).coalesce().to('cpu')

        return global_sparse_tensor

    def evaluate_chunks_directly(self):
        """
        Evaluate chunks directly without parallel computation.

        Returns:
            torch.sparse.FloatTensor: The evaluated sparse tensor.
        """
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
                
                chunk_dense_tensor = chunk_sparse_tensor.to_dense().to(self.device)
                func_output = self.function(self.mask_coef * chunk_dense_tensor)
                func_output_sparse = func_output.to_sparse().to('cpu').coalesce()

                add_indices = func_output_sparse.indices().to(dtyped) + torch.tensor(
                    [[chunk_start + self.eval_start_index]] + [[0]] * (func_output_sparse.indices().size(0) - 1), dtype=dtyped, device=torch.device('cpu')
                )

                global_storage['indices'].append(add_indices.cpu())
                global_storage['values'].append(func_output_sparse.values().cpu())

                del chunk_dense_tensor, chunk_sparse_tensor, func_output, func_output_sparse, add_indices
                torch.cuda.empty_cache()

            global_indices = torch.cat(global_storage['indices'], dim=1)
            global_values = torch.cat(global_storage['values'], dim=0)

            global_sparse_tensor = torch.sparse_coo_tensor(global_indices, global_values, size=self.output_size).coalesce().to('cpu')

        return global_sparse_tensor
    


@ray.remote
class SparseWorkerParallel:
    def __init__(self, x_chunk, device):
        self.x_chunk = x_chunk.to(device)
        self.device = device

    def copy(self,indice,indice_in=None):
        copy_name = f'copy_{indice}'
        
        if indice_in is None:
            copy_ = copy.deepcopy(self.x_chunk)
            setattr(self, copy_name, copy_)
        
        else:
            name_in = f'copy_{indice_in}'
            copy_=copy.deepcopy(getattr(self,name_in))
            setattr(self,copy_name,copy_)

    
    def add(self,indice_1=None,indice_2=None,result_indice=None):
            copy_name_1 = f'copy_{indice_1}'
            copy_name_2 = f'copy_{indice_2}'
            if hasattr(self, copy_name_1):
                x_1 = getattr(self, copy_name_1)
            if hasattr(self, copy_name_1):
                x_2 = getattr(self, copy_name_2)
            if indice_1 is None:
                x_1= self.x_chunk
            if indice_2 is None:
                x_2 = self.x_chunk
            x=(x_1+x_2).coalesce()
            if result_indice is None:
                self.x_chunk=x
            else :
                copy_name = f'copy_{result_indice}'
                setattr(self, copy_name, x)


    def sub(self,indice_1=None,indice_2=None,result_indice=None):
            copy_name_1 = f'copy_{indice_1}'
            copy_name_2 = f'copy_{indice_2}'
            if hasattr(self, copy_name_1):
                x_1 = getattr(self, copy_name_1)
            if hasattr(self, copy_name_1):
                x_2 = getattr(self, copy_name_2)
            if indice_1 is None:
                x_1= self.x_chunk
            if indice_2 is None:
                x_2 = self.x_chunk
            x=(x_1-x_2).coalesce()
            if result_indice is None:
                self.x_chunk=x
            else :
                copy_name = f'copy_{result_indice}'
                setattr(self, copy_name, x)


    def forward(self,function,mask_epsilon,chunk_size=1,over_copy = False, indice_copy=None):
        """
        Evaluate chunks of the sparse tensor.

        Returns:
            tuple: Indices and values of the evaluated sparse tensor.
        """
        if not over_copy:
            x = self.x_chunk
        else : 
            copy_name = f'copy_{indice_copy}'
            if hasattr(self, copy_name):
                x = getattr(self, copy_name)
                print(x.size())

        dense_shape = list(x.size())
     
        x_0 = torch.zeros(1, *dense_shape[1:])
    
        output_size = list(function(x_0).shape)
        output_size[0] = dense_shape[0]

        with torch.no_grad():
            indices = x.indices().t()
            values = x.values()

            global_storage = {
                'indices': [],
                'values': []
            }

            num_chunks = (x.size(0) + chunk_size - 1) // chunk_size
          

            for i in range(num_chunks):
                chunk_start = i * chunk_size
                chunk_end = min((i + 1) * chunk_size, x.size(0))
                mask = (indices[:, 0] >= chunk_start) & (indices[:, 0] < chunk_end)

                chunk_indices = indices[mask]
                chunk_indices[:, 0] -= chunk_start

                chunk_values = values[mask]
                chunk_size = chunk_end - chunk_start

                chunk_sparse_tensor = torch.sparse_coo_tensor(
                    chunk_indices.t(), chunk_values,
                    torch.Size([chunk_size] + dense_shape[1:])
                ).coalesce()

                chunk_dense_tensor = chunk_sparse_tensor.to_dense().to(self.device)
          
                func_output = function(mask_epsilon * chunk_dense_tensor)
           

                func_output_sparse = func_output.to_sparse().to('cpu').coalesce()
                add_indices = func_output_sparse.indices().to(dtyped) + torch.tensor(
                    [[chunk_start ]] + [[0]] * (func_output_sparse.indices().size(0) - 1), dtype=torch.long, device=torch.device('cpu')
                )

                global_storage['indices'].append(add_indices.cpu())
                global_storage['values'].append(func_output_sparse.values().cpu())

              
          

     

            
            global_indices = torch.cat(global_storage['indices'], dim=1)
            global_values = torch.cat(global_storage['values'], dim=0)
        if not over_copy:    
            self.x_chunk = torch.sparse_coo_tensor(global_indices, global_values, size=output_size).coalesce().to('cpu')
            return torch.sum(torch.abs(self.x_chunk),dim=0).coalesce().to_dense()
        else:
            x=torch.sparse_coo_tensor(global_indices, global_values, size=output_size).coalesce().to('cpu')
            setattr(self,copy_name,x)

            return torch.sum(torch.abs(x),dim=0).coalesce().to_dense()
    

class SparseEvaluationParallel:
    def __init__(self, x: FloatTensor, num_workers: int, device: torch.device):
        self.x = x
        self.num_workers = num_workers
        self.device = device
        self.dense_shape = list(x.size())
        
        self.workers = self._distribute_tensor()

    def _distribute_tensor(self):
        workers = []
        chunk_size = (self.x.size(0) + self.num_workers - 1) // self.num_workers
        indices = self.x.indices()
        values = self.x.values()
        indices = indices.t()
     
        for i in range(self.num_workers):
            chunk_start = i * chunk_size
            chunk_end = min((i + 1) * chunk_size, self.dense_shape[0])
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




            worker = SparseWorkerParallel.remote(worker_sparse_tensor, self.device)
            workers.append(worker)
        return workers
    
    def copy(self,indice,indice_in=None):
        for i in range(self.num_workers):
            ray.get(self.workers[i].copy.remote(indice = indice,indice_in=indice_in))

    def add(self,indice_1,indice_2,result_indice=None):
        for i in range(self.num_workers):
            ray.get(self.workers[i].add.remote(indice_1 = indice_1,indice_2=indice_2,result_indice=result_indice))
    def sub(self,indice_1,indice_2,result_indice=None):
        for i in range(self.num_workers):
            ray.get(self.workers[i].sub.remote(indice_1 = indice_1,indice_2=indice_2,result_indice=result_indice))
   

    def forward(self, mask: torch.Tensor, function: Callable,chunk_size:int=1,over_copy =False,indice_copy=None):
        tasks = []
        for i in range(self.num_workers):

            tasks.append(self.workers[i].forward.remote(function, mask.to(self.device),chunk_size,over_copy = over_copy, indice_copy=indice_copy))
        results = ray.get(tasks)
        sum=0
        for result in results: 
            sum += result   
        
        return sum

def test_sparse_evaluation(x):
    """
    Test the SparseEvaluation class with a sample input tensor.

    Args:
        x (torch.sparse.FloatTensor): The input sparse tensor.
    """
    function = nn.Conv2d(3, 3, 3)
    function.bias.data = torch.zeros_like(function.bias.data)
    eval = SparseEvaluation(x, 100, function=function, device=torch.device('cpu'))
    result = eval.evaluate_chunks_directly()
    sum_result = torch.sum(result, dim=0)
    with torch.no_grad():
        print(torch.sum(result) - torch.sum(function(x.to_dense())))
        print(f"diff de sum {sum_result.to_dense() - torch.sum(torch.abs(function(x.to_dense())), dim=0)}")

def test_sparse_evaluation_ray(x):
    """
    Test the SparseEvaluation class with parallel computation using Ray.

    Args:
        x (torch.sparse.FloatTensor): The input sparse tensor.
    """
    function = nn.Conv2d(3, 3, 3)
    function.bias.data = torch.zeros_like(function.bias.data)
    eval = SparseEvaluation(x, 100, function=function, device=torch.device('cpu'))
    result = eval.evaluate_all_chunks(num_workers=5)
    sum_result = torch.sum(result, dim=0)
    with torch.no_grad():
        print(torch.sum(result) - torch.sum(function(x.to_dense())))
        print(f"diff de sum {sum_result.to_dense() - torch.sum(torch.abs(function(x.to_dense())), dim=0)}")

if __name__ == "__main__":
    x = torch.randn(500, 3, 28, 28).to_sparse()
    test_sparse_evaluation(x)
    test_sparse_evaluation_ray(x)
