import sys
import argparse
import torch
import torch.nn as nn
from torch.sparse import FloatTensor
from typing import List, Union, Tuple, Callable

# Add paths for imports
sys.path.append('app/src')
sys.path.append('./src')


def list_of_shape(tensor: torch.Tensor) -> List[int]:
    """Returns the shape of a tensor as a list."""
    tensor = torch.tensor(tensor)
    return list(tensor.shape)

class ZonoSparseGeneration:
    """This class generates a sparse representation of an abstract domain."""
    
    def __init__(self, input: FloatTensor, noise_intensity : Union[float, torch.Tensor]=0, noise_type: str = 'additive', 
                 indices=None, from_trash=False, start_index=None):
        self.input = input.to('cpu')
        self.noise_intensity = torch.tensor(noise_intensity).to('cpu')
        self.noise_type = noise_type
        self.input_shape = list_of_shape(input)
        self.indices = indices
        self.from_trash = from_trash
        self.start_index = start_index

    def total_zono(self):
        """Generates a sparse zonotope."""
        if not self.from_trash:
            dim_input = torch.tensor(self.input_shape).numel()

            if dim_input == 1:
                global_storage = {'indices': [], 'values': []}

                if self.indices is None:
                    num_elements = self.input_shape[0]
                    self.indices = torch.arange(1, num_elements, 1)
                else:
                    self.indices = self.indices.to('cpu')
                    self.indices = torch.tensor(self.indices)
                    num_elements = self.indices.numel()

                if len(self.noise_intensity.flatten()) == 1:
                    self.noise_intensity = self.noise_intensity * torch.ones_like(self.indices)
                else:
                    assert self.noise_intensity.size() == self.indices.size(), 'the length of noise intensity must be one or equal to indices shape'

                for i in range(num_elements):
                    global_storage['indices'].append([self.indices[i], self.indices[i]])
                    global_storage['values'].append(self.noise_intensity[i])

                indice_tensor = torch.tensor(global_storage['indices'], dtype=torch.int32).t()
                values_tensor = torch.tensor(global_storage['values'], dtype=torch.float32)
                sparse_zonotope = torch.sparse_coo_tensor(indice_tensor, values_tensor, size=(self.input_shape[0], self.input_shape[0])).coalesce()

                return self.input, sparse_zonotope.to_dense()

            if dim_input == 2:
                self.input = self.input.unsqueeze(0)
                self.input_shape = list_of_shape(self.input)
            
            if dim_input == 4:
                self.input = self.input.squeeze(0)
                print("WARNING: Trying to generate abstract Sparse tensor from a batch, only the first element will be used")
                self.input_shape = list_of_shape(self.input)

            if self.indices is None:
                assert len(self.noise_intensity.flatten()) == 1, 'Shape of noise and indices do not match'
                num_elements = self.input_shape[0]
                self.indices = torch.arange(1, num_elements, 1)
                global_storage = {'indices': [], 'values': []}
                num_elements = self.input_shape[0] * self.input_shape[1] * self.input_shape[2]

                for i in range(num_elements):
                    dim_3 = i // (self.input_shape[1] * self.input_shape[2])
                    rem = i % (self.input_shape[1] * self.input_shape[2])
                    dim_1 = rem // self.input_shape[1]
                    dim_2 = rem % self.input_shape[2]
                    global_storage['indices'].append([i, dim_3, dim_1, dim_2])
                    global_storage['values'].append(self.noise_intensity)

                indice_tensor = torch.tensor(global_storage['indices'], dtype=torch.int32).t()
                values_tensor = torch.tensor(global_storage['values'], dtype=torch.float32)
                sparse_zonotope = torch.sparse_coo_tensor(indice_tensor, values_tensor, size=(num_elements, self.input_shape[0], self.input_shape[1], self.input_shape[2])).coalesce()

            else:
                self.indices = torch.tensor(self.indices).to('cpu')
                assert len(self.indices) == len(self.noise_intensity), 'Length of Noise_intensity and indices mismatch'
                global_storage = {'indices': [], 'values': []}
                num_elements = len(self.indices)

                for i in range(num_elements):
                    if len(self.indices[i]) == 2:
                        global_storage['indices'].append(torch.cat((torch.tensor([i, 0]), self.indices[i])).tolist())
                    else:
                        global_storage['indices'].append(torch.cat((torch.tensor([i]), self.indices[i])).tolist())
                    global_storage['values'].append(self.noise_intensity[i])

                indice_tensor = torch.tensor(global_storage['indices'], dtype=torch.int32).t()
                values_tensor = torch.tensor(global_storage['values'], dtype=torch.float32)
                print(indice_tensor)
                print(values_tensor)

                sparse_zonotope = torch.sparse_coo_tensor(indice_tensor, values_tensor, size=(num_elements, self.input_shape[0], self.input_shape[1], self.input_shape[2])).coalesce()

            return self.input, sparse_zonotope

        if self.from_trash:
            if not self.start_index:
                print('Warning, start_index is 0, should start at the depth of abstract domain')
                self.start_index = 0

            global_storage = {'indices': [], 'values': []}
            indices = torch.nonzero(self.input)
            if len(indices)==0: 
                return self.input, None
          

            for i, indice in enumerate(indices):
                sparse_indice = torch.cat((torch.tensor([i + self.start_index]), indice[1:])).tolist()
                global_storage['indices'].append(sparse_indice)
                global_storage['values'].append(self.input[tuple(indice.tolist())])

            indice_tensor = torch.tensor(global_storage['indices'], dtype=torch.int32).t()
            values_tensor = torch.tensor(global_storage['values'], dtype=torch.float32)
            dim = tuple(torch.cat((torch.tensor([len(indices)]), torch.tensor(list_of_shape(self.input.squeeze(0))))))

            sparse_zonotope = torch.sparse_coo_tensor(indice_tensor, values_tensor, size=dim).coalesce()

            return self.input, sparse_zonotope
def main(args):
    main()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sparse zonotope representations.")

    # Define arguments
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size of the input tensor.')
    parser.add_argument('--height', type=int, default=3, help='Height of the input tensor.')
    parser.add_argument('--width', type=int, default=3, help='Width of the input tensor.')
    parser.add_argument('--noise_intensity', type=float, nargs='+', default=[1.0], help='Noise intensity values.')
    parser.add_argument('--indices', type=int, nargs='+', help='Indices for the sparse representation.')
    parser.add_argument('--from_trash', action='store_true', help='Whether to use the from_trash option.')
    parser.add_argument('--start_index', type=int, default=0, help='Starting index for the from_trash option.')

    # Parse arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args)
