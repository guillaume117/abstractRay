import torch
import torch.nn.functional as F
from torch.sparse import FloatTensor
from typing import Callable
import torch.nn as nn
import ray
from tqdm import tqdm
import os

os.environ["RAY_NUM_CPUS"] = str(os.cpu_count())

dtyped = torch.long

@ray.remote(num_gpus=1)
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

            if global_storage['indices']:
                global_indices = torch.cat(global_storage['indices'], dim=1)
                global_values = torch.cat(global_storage['values'], dim=0)
            else:
                global_indices = torch.empty((2, 0), dtype=dtyped)
                global_values = torch.empty((0,), dtype=torch.float32)

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

            if self.device == torch.device('cuda'):
                self.num_gpus = 1
                self.num_cpus = os.cpu_count()
            else:
                self.num_gpus = 0
                self.num_cpus = os.cpu_count()

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

            if global_indices:
                global_indices = torch.cat(global_indices, dim=1)
                global_values = torch.cat(global_values, dim=0)
            else:
                global_indices = torch.empty((2, 0), dtype=dtyped)
                global_values = torch.empty((0,), dtype=torch.float32)

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
