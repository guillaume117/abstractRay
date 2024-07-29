import torch
import torch.nn as nn
from torch.sparse import FloatTensor
from typing import Callable
import ray
dtyped = torch.long

@ray.remote
class SparseWorkerParallel:
    def __init__(self, x_chunk, device):
        self.x_chunk = x_chunk.to(device)
        self.device = device


    

    def forward(self,function,mask_epsilon,chunk_size=1):
        """
        Evaluate chunks of the sparse tensor.

        Returns:
            tuple: Indices and values of the evaluated sparse tensor.
        """
        dense_shape = list(self.x_chunk.size())
     
        x_0 = torch.zeros(1, *dense_shape[1:])
    
        output_size = list(function(x_0).shape)
        output_size[0] = dense_shape[0]

        with torch.no_grad():
            indices = self.x_chunk.indices().t()
            values = self.x_chunk.values()

            global_storage = {
                'indices': [],
                'values': []
            }

            num_chunks = (self.x_chunk.size(0) + chunk_size - 1) // chunk_size
          

            for i in range(num_chunks):
                chunk_start = i * chunk_size
                chunk_end = min((i + 1) * chunk_size, self.x_chunk.size(0))
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
        self.x_chunk = torch.sparse_coo_tensor(global_indices, global_values, size=output_size).coalesce().to('cpu')
     

        return torch.sum(torch.abs(self.x_chunk),dim=0).coalesce().to_dense()


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

    def forward(self, mask: torch.Tensor, function: Callable,chunk_size:int =1):
        tasks = []
        for i in range(self.num_workers):
            tasks.append(self.workers[i].forward.remote(function, mask.to(self.device)))
        results = ray.get(tasks)
        sum=0
        for result in results: 
            sum += result   
        
        return sum
# Initialize Ray
ray.init(ignore_reinit_error=True)

# Example usage
test = torch.ones(30, 2, 3, 3).to_sparse()
evaluator = SparseEvaluationParallel(test, 3, torch.device('cpu'))

result = evaluator.forward(torch.ones(1, 2, 3, 3), nn.Identity())
print(result)

result = evaluator.forward(2.01*torch.ones(1, 2, 3, 3), nn.Identity())
print(result)

result = evaluator.forward(torch.ones(1, 2, 3, 3), nn.Identity())
print(result)
result = evaluator.forward(torch.ones(1, 2, 3, 3), nn.Identity())
print(result)
