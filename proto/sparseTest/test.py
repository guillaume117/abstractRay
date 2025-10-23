import torch
import numpy as np
from pybind11 import get_include
from torch.utils.cpp_extension import load

# Load CUDA extension
sparse_conv = load(
    name="sparse_conv",
    sources=["sparse_conv.cpp", "sparse_conv.cu"],
    extra_include_paths=[get_include()],
    verbose=True
)

class SparseTensor:
    def __init__(self, data, shape):
        """
        :param data: List of dictionaries where each dictionary contains {coordinate: (batch_index, value)} pairs.
        :param shape: Tuple defining the shape of the tensor (batch_size, channels, height, width).
        """
        self.data = data
        self.shape = shape

    def to_dense(self):
        """Convert sparse representation to dense tensor."""
        dense = torch.zeros(self.shape, device="cuda")
        for coord, (batch_idx, value) in self.data.items():
            dense[batch_idx, :, coord[0], coord[1]] = value
        return dense

    @staticmethod
    def from_dense(tensor):
        """Convert dense tensor to sparse representation."""
        data = {}
        tensor = tensor.cpu()
        for batch_idx in range(tensor.shape[0]):
            for c in range(tensor.shape[2]):
                for h in range(tensor.shape[3]):
                    for w in range(tensor.shape[4]):
                        value = tensor[batch_idx, :, h, w]
                        if value != 0:
                            data[(h, w)] = (batch_idx, value.item())
        return SparseTensor(data, tensor.shape)


def sparse_convolution_2d(conv2d_layer, sparse_tensor):
    """
    Perform convolution operation on SparseTensor using torch.nn.Conv2d weights.
    :param conv2d_layer: Instance of torch.nn.Conv2d.
    :param sparse_tensor: Instance of SparseTensor.
    :return: SparseTensor representing the result.
    """
    assert isinstance(conv2d_layer, torch.nn.Conv2d), "conv2d_layer must be an instance of torch.nn.Conv2d"
    assert isinstance(sparse_tensor, SparseTensor), "sparse_tensor must be an instance of SparseTensor"

    # Extract weights from Conv2d layer
    weights = conv2d_layer.weight.detach().cuda()

    # Convert SparseTensor to dense representation
    dense_input = sparse_tensor.to_dense()

    # Perform CUDA-based sparse convolution (custom kernel in sparse_conv.cu)
    output_dense = sparse_conv.forward(dense_input, weights, conv2d_layer.stride, conv2d_layer.padding)

    # Convert output dense tensor back to SparseTensor
    output_sparse = SparseTensor.from_dense(output_dense)
    return output_sparse

# Example Usage
if __name__ == "__main__":
    # Define a Conv2d layer
    conv_layer = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False).cuda()

    # Example SparseTensor
    example_data = {
        (0, 0): (0, 1.0),  # Example: Batch index 0, channel 0, value 1.0
        (1, 1): (1, 2.0),
    }
    sparse_input = SparseTensor(example_data, (2, 3, 5, 5))  # Shape: (Batch=2, Channels=3, Height=5, Width=5)

    # Perform sparse convolution
    result_sparse_tensor = sparse_convolution_2d(conv_layer, sparse_input)
    print("Output Shape:", result_sparse_tensor.shape)
    print("Output Data:", result_sparse_tensor.data)
