import torch
import matplotlib.pyplot as plt

class ZonoSparseGeneration:
    """
    A class for generating sparse zonotopes from tensors and noise levels.

    This class provides methods to create zonotopes based on input tensors, noise levels, and specific regions of interest.

    Attributes:
        global_storage (dict): A dictionary to store global indices and values.
    """
    def __init__(self):
        self.global_storage = {'indices': [], 'values': []}

    def zono_from_input_noise_level_and_mask(self, tensor_input, x_min, x_max, y_min, y_max, noise_level):
        """
        Generate a zonotope from an input tensor, noise level, and specified mask region.

        Args:
            tensor_input (torch.Tensor): The input tensor.
            x_min (int): Minimum x-coordinate of the mask.
            x_max (int): Maximum x-coordinate of the mask.
            y_min (int): Minimum y-coordinate of the mask.
            y_max (int): Maximum y-coordinate of the mask.
            noise_level (float): The noise level to apply.

        Returns:
            torch.sparse.FloatTensor: The generated zonotope.
        """
        output = noise_level * torch.zeros_like(tensor_input)
        output[:, :, x_min:x_max, y_min:y_max] = noise_level
        zonotope = self.zono_from_tensor(output)
        return zonotope.coalesce()

    def zono_from_tensor(self, noise_intensity, start_index=0):
        """
        Generate a zonotope from a tensor.

        Args:
            noise_intensity (torch.Tensor): The tensor representing noise intensity.
            start_index (int, optional): The starting index for the zonotope. Defaults to 0.

        Returns:
            torch.sparse.FloatTensor: The generated zonotope.
        """
        assert noise_intensity.size(0) == 1, 'First dimension size must be 1'
        noise_intensity = noise_intensity.to_sparse().coalesce()
        noise_intensity.indices()[0] = torch.arange(noise_intensity._nnz()) + start_index
        size = (noise_intensity._nnz() + start_index, *noise_intensity.size()[1:])
        zonotope = torch.sparse_coo_tensor(noise_intensity.indices(), noise_intensity.values(), size=size)
        return zonotope.coalesce()

    def zono_from_noise_level_and_tensor(self, noise_level, tensor):
        """
        Generate a zonotope from a noise level and tensor.

        Args:
            noise_level (float): The noise level to apply.
            tensor (torch.Tensor): The input tensor.

        Returns:
            torch.sparse.FloatTensor: The generated zonotope.
        """
        noise_intensity = noise_level * torch.ones_like(tensor)
        zonotope = self.zono_from_tensor(noise_intensity=noise_intensity)
        return zonotope.coalesce()

if __name__ == "__main__":
    test = ZonoSparseGeneration().zono_from_input_noise_level_and_mask(torch.randn(1, 3, 52, 52), 10, 15, 10, 15, 1)
    print(test)
    plt.imshow(torch.sum(test.coalesce(), dim=0).to_dense().permute(1, 2, 0))
    plt.show()
