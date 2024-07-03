import ray
import torch
import torch.nn as nn
from torch.sparse import FloatTensor
from typing import Callable, Tuple
from tqdm import tqdm

@ray.remote(num_gpus=1)
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
        indices_x = self.x_chunk.indices().t()
        values_x = self.x_chunk.values()

        indices_y = self.y_chunk.indices().t()
        values_y = self.y_chunk.values()

        global_storage = {
            'indices': [],
            'values': []
        }

        num_chunks = (self.x_chunk.shape[0] + self.chunk_size - 1) // self.chunk_size

        function_sum = None

        for i in range(num_chunks):
            with torch.no_grad():
                chunk_start = i * self.chunk_size
                chunk_end = min((i + 1) * self.chunk_size, self.x_chunk.shape[0])

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
                    func_output = torch.cat((chunk_dense_x, chunk_dense_y), dim=1)

                func_sum = torch.abs(func_output).sum(dim=0)
                if function_sum is None:
                    function_sum = func_sum
                else:
                    function_sum += func_sum

                func_output_sparse = func_output.to_sparse().to('cpu').coalesce()
                add_indices = func_output_sparse.indices().to(torch.int32) + torch.tensor(
                    [[chunk_start + self.worker_start_index]] + [[0]] * (func_output_sparse.indices().size(0) - 1), dtype=torch.int32, device=torch.device('cpu ')
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

class SparseAddition:
    def __init__(self, x: FloatTensor, y: FloatTensor, chunk_size: int, device=torch.device('cpu')):
        max_batch_size = max(x.shape[0], y.shape[0])
        self.common_size = [max_batch_size] + list(x.size()[1:])
        self.x = x.coalesce()
        self.y = y.coalesce()
        self._adjust_tensor_size()

        self.chunk_size = chunk_size
        self.dense_shape = self.common_size
        self.device = device

    def _adjust_tensor_size(self):
        if self.x.shape[0] <= self.common_size[0]:
            self.x = torch.sparse_coo_tensor(self.x.indices(), self.x.values(), size=self.common_size).coalesce()
        else:
            self.y = torch.sparse_coo_tensor(self.y.indices(), self.y.values(), size=self.common_size).coalesce()

    def addition(self, num_workers):
        return self._operate('addition', num_workers)

    def substraction(self, num_workers):
        return self._operate('substraction', num_workers)
    
    def concat(self, num_workers):
        return self._operate('concat', num_workers)

    def _operate(self, operation, num_workers):
        if num_workers == 0:
            return self._process_chunks_directly(operation)

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

            worker = SparseWorker.remote(worker_sparse_tensor_x, worker_sparse_tensor_y, chunk_size=self.chunk_size, operation=operation, dense_shape=self.dense_shape, worker_start_index=chunk_start, device=self.device)
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

        return global_sparse_tensor.to('cpu'), function_sum.to('cpu')

    def _process_chunks_directly(self, operation):
        indices_x = self.x.indices().t()
        values_x = self.x.values()

        indices_y = self.y.indices().t()
        values_y = self.y.values()

        global_storage = {
            'indices': [],
            'values': []
        }

        num_chunks = (self.x.shape[0] + self.chunk_size - 1) // self.chunk_size

        function_sum = None

        for i in tqdm(range(num_chunks)):
            with torch.no_grad():
                chunk_start = i * self.chunk_size
                chunk_end = min((i + 1) * self.chunk_size, self.x.shape[0])

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

                if operation == 'addition':
                    func_output = chunk_dense_x + chunk_dense_y
                elif operation == 'substraction':
                    func_output = chunk_dense_x - chunk_dense_y
                elif operation == 'concat':
                    func_output = torch.cat((chunk_dense_x, chunk_dense_y), dim=1)

                func_sum = torch.abs(func_output).sum(dim=0)
                if function_sum is None:
                    function_sum = func_sum
                else:
                    function_sum += func_sum

                func_output_sparse = func_output.to_sparse().to('cpu').coalesce()
                add_indices = func_output_sparse.indices().to(torch.int32) + torch.tensor(
                    [[chunk_start]] + [[0]] * (func_output_sparse.indices().size(0) - 1), dtype=torch.int32, device=torch.device('cpu')
                )

                global_storage['indices'].append(add_indices.cpu())
                global_storage['values'].append(func_output_sparse.values().cpu())

                del chunk_dense_x, chunk_dense_y, chunk_sparse_tensor_x, chunk_sparse_tensor_y, func_output, func_output_sparse, add_indices
                torch.cuda.empty_cache()

        global_indices = torch.cat(global_storage['indices'], dim=1)
        global_values = torch.cat(global_storage['values'], dim=0)

        global_sparse_tensor = torch.sparse_coo_tensor(global_indices, global_values, size=torch.Size(self.common_size)).coalesce().to('cpu')

        return global_sparse_tensor.to('cpu'), function_sum.to('cpu')
