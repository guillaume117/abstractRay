import torch
import torch.nn.functional as F
from torch.sparse import FloatTensor
from typing import Callable
import torch.nn as nn
import ray
from tqdm import tqdm

@ray.remote(num_gpus=1)
class SparseWorker:
    def __init__(self, x_chunk, chunk_size, mask_coef, function, dense_shape, worker_start_index, device):
        self.x_chunk = x_chunk.coalesce().to(device)
        self.chunk_size = chunk_size
        self.mask_coef = mask_coef.to(device)
        if isinstance(function, nn.Module):
            function = function.to(device)
        self.function = function
        self.dense_shape = dense_shape
        self.worker_start_index = worker_start_index
        self.device = device

    def evaluate_chunks(self):
        indices = self.x_chunk.indices().t()
        values = self.x_chunk.values()

        global_storage = {
            'indices': [],
            'values': []
        }
        function_sum = None

        num_chunks = (self.x_chunk.size(0) + self.chunk_size - 1) // self.chunk_size

        for i in range(num_chunks):
            with torch.no_grad():
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

                func_sum = torch.abs(func_output).sum(dim=0)
                if function_sum is None:
                    function_sum = func_sum
                else:
                    function_sum += func_sum

                func_output_sparse = func_output.to_sparse().coalesce()
                add_indices = func_output_sparse.indices().to(torch.int32).to('cpu') + torch.tensor(
                    [[chunk_start + self.worker_start_index]] + [[0]] * (func_output_sparse.indices().size(0) - 1), dtype=torch.int32, device=torch.device('cpu')
                )

                global_storage['indices'].append(add_indices.cpu())
                global_storage['values'].append(func_output_sparse.values().cpu())

                del chunk_dense_tensor, chunk_sparse_tensor, func_output, func_output_sparse, add_indices
                torch.cuda.empty_cache()
                progress = (i + 1) / num_chunks
                if i % 10 == 0:
                    print(f"Worker {self.worker_start_index // self.dense_shape[0]} progress: {progress:.2%}")

        global_indices = torch.cat(global_storage['indices'], dim=1)
        global_values = torch.cat(global_storage['values'], dim=0)

        return global_indices, global_values, function_sum.to('cpu')

class SparseEvaluation:
    def __init__(self, x: FloatTensor, chunk_size: int, mask_coef: FloatTensor = None, function: Callable = None, eval_start_index=0, device=torch.device('cpu')):
        self.x = x
        self.chunk_size = chunk_size
        self.dense_shape = list(x.size())
        self.device = device
        self.eval_start_index = eval_start_index
        

        if function is None:
            self.function = lambda x: x
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
        print("output_size", self.output_size)

    def evaluate_all_chunks(self, num_workers):
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

            worker_sparse_tensor = torch.sparse_coo_tensor(
                worker_indices.t(), worker_values,
                torch.Size([worker_size] + self.dense_shape[1:])
            ).coalesce()

            worker = SparseWorker.remote(worker_sparse_tensor, self.chunk_size, self.mask_coef, self.function, self.dense_shape, chunk_start, self.device)
            workers.append(worker.evaluate_chunks.remote())

        results = ray.get(workers)
        global_indices = []
        global_values = []
        function_sum = None

        for add_indices, func_values, func_sum in results:
            add_indices = add_indices + torch.tensor(
                [[self.eval_start_index]] + [[0]] * (add_indices.size(0) - 1), dtype=torch.int32, device=torch.device('cpu')
            )
            global_indices.append(add_indices)
            global_values.append(func_values)
            if function_sum is None:
                function_sum = func_sum
            else:
                function_sum += func_sum

        global_indices = torch.cat(global_indices, dim=1)
        global_values = torch.cat(global_values, dim=0)

        global_sparse_tensor = torch.sparse_coo_tensor(global_indices, global_values, size=self.output_size).coalesce().to('cpu')

        return global_sparse_tensor, function_sum.to('cpu')

    def evaluate_chunks_directly(self):
        indices = self.x.indices().t()
        values = self.x.values()
        self.function = self.function.to(self.device)

        global_storage = {
            'indices': [],
            'values': []
        }
        function_sum = None

        num_chunks = (self.x.size(0) + self.chunk_size - 1) // self.chunk_size

        for i in tqdm(range(num_chunks)):
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

                chunk_dense_tensor = chunk_sparse_tensor.to_dense().to(self.device)
                func_output = self.function(self.mask_coef * chunk_dense_tensor)

                func_sum = torch.abs(func_output).sum(dim=0)
                if function_sum is None:
                    function_sum = func_sum
                else:
                    function_sum += func_sum

                func_output_sparse = func_output.to_sparse().coalesce()
                add_indices = func_output_sparse.indices().to(torch.int32).to('cpu') + torch.tensor(
                    [[chunk_start + self.eval_start_index]] + [[0]] * (func_output_sparse.indices().size(0) - 1), dtype=torch.int32, device=torch.device('cpu')
                )

                global_storage['indices'].append(add_indices.cpu())
                global_storage['values'].append(func_output_sparse.values().cpu())

                del chunk_dense_tensor, chunk_sparse_tensor, func_output, func_output_sparse, add_indices
                torch.cuda.empty_cache()

        global_indices = torch.cat(global_storage['indices'], dim=1)
        global_values = torch.cat(global_storage['values'], dim=0)

        global_sparse_tensor = torch.sparse_coo_tensor(global_indices, global_values, size=self.output_size).coalesce().to('cpu')

        return global_sparse_tensor, function_sum.to('cpu')
