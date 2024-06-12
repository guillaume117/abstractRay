import ray
import torch
import torch.nn.functional as F
from torch.sparse import FloatTensor
from typing import Callable

device = torch.device("cpu")


@ray.remote(num_cpus=2, memory=4*1024*1024*1024) 
class SparseWorker:
    def __init__(self, x_chunk, chunk_size, mask_coef, function, dense_shape, global_start_index):
        self.x_chunk = x_chunk.coalesce()
        self.chunk_size = chunk_size
        self.mask_coef = mask_coef
        self.function = function
        self.dense_shape = dense_shape
        self.global_start_index = global_start_index

    def evaluate_chunks(self):
        indices = self.x_chunk.indices().t()
        values = self.x_chunk.values()

        global_storage = {
            'indices': [],
            'values': []
        }
        function_sum = None

        num_chunks = (self.dense_shape[0] + self.chunk_size - 1) // self.chunk_size

        for i in range(num_chunks):
            
            with torch.no_grad():
                chunk_start = i * self.chunk_size
                chunk_end = min((i + 1) * self.chunk_size, self.dense_shape[0])
                mask = (indices[:, 0] >= chunk_start) & (indices[:, 0] < chunk_end)

                chunk_indices = indices[mask]
                chunk_indices[:, 0] -= chunk_start

                chunk_values = values[mask]
                chunk_size = chunk_end - chunk_start

                chunk_sparse_tensor = torch.sparse_coo_tensor(
                    chunk_indices.t(), chunk_values,
                    torch.Size([chunk_size] + self.dense_shape[1:])
                ).coalesce()

                chunk_dense_tensor = chunk_sparse_tensor.to_dense().to(device)
                func_output = self.function(self.mask_coef * chunk_dense_tensor)

                func_sum = torch.abs(func_output).sum(dim=0)
                if function_sum is None:
                    function_sum = func_sum
                else:
                    function_sum += func_sum

                func_output_sparse = func_output.to_sparse().coalesce()
                add_indices = func_output_sparse.indices().to(torch.int32) + torch.tensor(
                    [[chunk_start + self.global_start_index], [0], [0], [0]], dtype=torch.int32, device=device
                )

                global_storage['indices'].append(add_indices.cpu())
                global_storage['values'].append(func_output_sparse.values().cpu())

                del chunk_dense_tensor, chunk_sparse_tensor, func_output, func_output_sparse, add_indices
                torch.cuda.empty_cache()

        global_indices = torch.cat(global_storage['indices'], dim=1)
        global_values = torch.cat(global_storage['values'], dim=0)

        return global_indices, global_values, function_sum


class SparseEvaluation:
    def __init__(self, x: FloatTensor, chunk_size: int, mask_coef: FloatTensor = None, function: Callable = None):
        self.x = x
        self.chunk_size = chunk_size
        self.dense_shape = list(x.size())
        
        if function is None:
            self.function = lambda x: x
        else:
            self.function = function

        x0 = torch.zeros(1, self.dense_shape[1], self.dense_shape[2], self.dense_shape[3]).to(device)
        
        if mask_coef is None:
            self.mask_coef = torch.ones_like(x0)
        else:
            self.mask_coef = mask_coef

        self.num_chunks = (self.dense_shape[0] + self.chunk_size - 1) // self.chunk_size
        self.output_size = list(self.function(x0).shape)
        self.output_size[0] = self.dense_shape[0]

    def evaluate_all_chunks(self, num_workers):
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

            worker_sparse_tensor = torch.sparse_coo_tensor(
                worker_indices.t(), worker_values,
                torch.Size([worker_size] + self.dense_shape[1:])
            ).coalesce()

            worker = SparseWorker.remote(worker_sparse_tensor, self.chunk_size, self.mask_coef, self.function, self.dense_shape, chunk_start)
            workers.append(worker.evaluate_chunks.remote())

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

        global_sparse_tensor = torch.sparse_coo_tensor(global_indices, global_values, size=self.output_size).coalesce()

    
        return global_sparse_tensor, function_sum


