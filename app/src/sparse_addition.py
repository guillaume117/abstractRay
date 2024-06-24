import ray
import torch
import torch.nn as nn
from torch.sparse import FloatTensor
from typing import Callable, Tuple

@ray.remote
class SparseWorker:
    def __init__(self, x_chunk, y_chunk, chunk_size, operation, dense_shape, worker_start_index, device):
        self.x_chunk = x_chunk.coalesce().to(device)
        self.y_chunk = y_chunk.coalesce().to(device)
        self.chunk_size = chunk_size
        self.operation = operation
        self.dense_shape = dense_shape
        self.worker_start_index = worker_start_index
        self.device = device

    def process_chunk(self):
        #try:
            indices_x = self.x_chunk.indices().t()
            values_x = self.x_chunk.values()

            indices_y = self.y_chunk.indices().t()
            values_y = self.y_chunk.values()


            global_storage = {
                'indices': [],
                'values': []
            }
     
            num_chunks = (self.x_chunk.size(0) + self.chunk_size - 1) // self.chunk_size
         
            function_sum = None

            for i in range(num_chunks):
                with torch.no_grad():
             
                    chunk_start = i * self.chunk_size
                  
                    chunk_end = min((i + 1) * self.chunk_size, self.x_chunk.size(0))
           

                    mask_x = (indices_x[:, 0] >= chunk_start) & (indices_x[:, 0] < chunk_end)
                    chunk_indices_x = indices_x[mask_x]
                    chunk_indices_x[:, 0] -= chunk_start
                    chunk_values_x = values_x[mask_x]
                    chunk_size_x = chunk_end - chunk_start

                    mask_y = (indices_y[:, 0] >= chunk_start) & (indices_y[:, 0] < chunk_end)
                    chunk_indices_y = indices_y[mask_y]
                    chunk_indices_y[:, 0] -= chunk_start
                    chunk_values_y = values_y[mask_y]
                    chunk_size_y = chunk_end - chunk_start

                    chunk_sparse_tensor_x = torch.sparse_coo_tensor(
                        chunk_indices_x.t(), chunk_values_x,
                        torch.Size([chunk_size_x] + self.dense_shape[1:])
                    ).coalesce()

                    chunk_sparse_tensor_y = torch.sparse_coo_tensor(
                        chunk_indices_y.t(), chunk_values_y,
                        torch.Size([chunk_size_y] + self.dense_shape[1:])
                    ).coalesce()

                    chunk_dense_x = chunk_sparse_tensor_x.to_dense().to(self.device)
                    chunk_dense_y = chunk_sparse_tensor_y.to_dense().to(self.device)

                    if self.operation == 'addition':
                        func_output = chunk_dense_x + chunk_dense_y
                    elif self.operation == 'substraction':
                        func_output = chunk_dense_x - chunk_dense_y
                    elif self.operation == 'concat':
                        func_output =  torch.cat((chunk_dense_x,chunk_dense_y),dim=1)
                    func_sum = torch.abs(func_output).sum(dim=0)
                    if function_sum is None:
                        function_sum = func_sum
                    else:
                        function_sum += func_sum

                    func_output_sparse = func_output.to_sparse().coalesce()
                    add_indices = func_output_sparse.indices().to(torch.int32) + torch.tensor(
                        [[chunk_start + self.worker_start_index]] + [[0]] * (func_output_sparse.indices().size(0) - 1), dtype=torch.int32, device=self.device
                    )

                    global_storage['indices'].append(add_indices.cpu())
                    global_storage['values'].append(func_output_sparse.values().cpu())

                    del chunk_dense_x, chunk_dense_y, chunk_sparse_tensor_x, chunk_sparse_tensor_y, func_output, func_output_sparse, add_indices
                    torch.cuda.empty_cache()
                    progress = (i + 1) / num_chunks
                    if i % 10 == 0:
                        print(f"Worker {self.worker_start_index // self.dense_shape[0]} progress: {progress:.2%}")

            global_indices = torch.cat(global_storage['indices'], dim=1)
            global_values = torch.cat(global_storage['values'], dim=0)

            return global_indices, global_values, function_sum
        #except Exception as e:
         #   return e

class SparseAddition:
    def __init__(self, x: FloatTensor, y: FloatTensor, chunk_size: int, device=torch.device('cpu')):

        # Ajuster les dimensions des tenseurs d'origine pour qu'elles soient égales à la plus grande dimension
        max_batch_size = max(x.shape[0], y.shape[0])
        self.common_size = [max_batch_size] + list(x.size()[1:])
        self.x = x.coalesce()
        self.y = y.coalesce()
        self._adjust_tensor_size()
        

        self.chunk_size = chunk_size
        self.dense_shape = self.common_size
        self.device = device

    def _adjust_tensor_size(self) -> FloatTensor:
        if self.x.shape[0] <= self.common_size[0]:
            self.x.size = self.common_size
        else:
            self.y.size = self.common_size
            
       

    def addition(self, num_workers):
        return self._operate('addition', num_workers)

    def substraction(self, num_workers):
        return self._operate('substraction', num_workers)
    
    def concat(self, num_workers):
        return self._operate('concat',num_workers)

    def _operate(self, operation, num_workers):
        indices_x = self.x.indices()
        values_x = self.x.values()
        indices_x = indices_x.t()

        indices_y = self.y.indices()
        values_y = self.y.values()
        indices_y = indices_y.t()

        max_batch_size = self.common_size[0]

        chunk_size_per_worker = (max_batch_size + num_workers - 1) // num_workers
        workers = []

        for i in range(num_workers):
            chunk_start = i * chunk_size_per_worker
            chunk_end = min((i + 1) * chunk_size_per_worker, max_batch_size)
            print(f'chunk start / end for worker {i} = {chunk_start} / {chunk_end}')

            mask_x = (indices_x[:, 0] >= chunk_start) & (indices_x[:, 0] < chunk_end)
            worker_indices_x = indices_x[mask_x]
            worker_indices_x[:, 0] -= chunk_start
            worker_values_x = values_x[mask_x]
            worker_size_x = chunk_end - chunk_start
            worker_sparse_tensor_x = torch.sparse_coo_tensor(
                worker_indices_x.t(), worker_values_x,
                torch.Size([worker_size_x] + self.dense_shape[1:])
            ).coalesce()

            mask_y = (indices_y[:, 0] >= chunk_start) & (indices_y[:, 0] < chunk_end)
            worker_indices_y = indices_y[mask_y]
            worker_indices_y[:, 0] -= chunk_start
            worker_values_y = values_y[mask_y]
            worker_size_y = chunk_end - chunk_start
            worker_sparse_tensor_y = torch.sparse_coo_tensor(
                worker_indices_y.t(), worker_values_y,
                torch.Size([worker_size_y] + self.dense_shape[1:])
            ).coalesce()

            worker = SparseWorker.remote(worker_sparse_tensor_x, worker_sparse_tensor_y,chunk_size= self.chunk_size,operation=operation,dense_shape= self.dense_shape,worker_start_index= chunk_start, device =self.device)
            workers.append(worker.process_chunk.remote())

        results = ray.get(workers)
        global_indices = []
        global_values = []
        function_sum = None

        for add_indices, func_values, func_sum in results:
            global_indices.append(add_indices)
            global_values.append(func_values)
            if function_sum is None:
                function_sum = func_sum
            else:
                function_sum += func_sum

        global_indices = torch.cat(global_indices, dim=1)
        global_values = torch.cat(global_values, dim=0)

        global_sparse_tensor = torch.sparse_coo_tensor(global_indices, global_values, size=torch.Size(self.common_size)).coalesce().to('cpu')

        return global_sparse_tensor, function_sum
"""
#tensor1 = torch.sparse.FloatTensor(torch.randint(0, 10, (4, 20)), torch.randn(20), (10, 3, 64, 64))
#tensor2 = torch.sparse.FloatTensor(torch.randint(0, 12, (4, 30)), torch.randn(30), (12, 3, 64, 64))
tensor1 = torch.randn(10,3,64,64).to_sparse_coo()
tensor2 = torch.randn(10,3,64,64).to_sparse_coo()

sparse_addition = SparseAddition(tensor1, tensor2, chunk_size=1, device=torch.device('cpu'))
result_add, sum1 = sparse_addition.addition(num_workers=2)
result_sub ,sum2 = sparse_addition.substraction(num_workers=2)



print(result_add)
print(result_sub)
print(sum1.size)
print("addition test failed if different from 0",torch.sum(result_add.to_dense()-(tensor1+tensor2)))
print("substraction test failed if different from 0",torch.sum(result_sub.to_dense()-(tensor1-tensor2)))

"""