import torch
import torch.nn.functional as F
import numpy as np



class SparseConv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, weights=None, bias_val=None):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize weights and biases
        if weights is None:
            self.weights = torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        else:
            self.weights = torch.tensor(weights, dtype=torch.float32)
        
        if bias:
            if bias_val is None:
                self.bias = torch.randn(out_channels)
            else:
                self.bias = torch.tensor(bias_val, dtype=torch.float32)
        else:
            self.bias = None

    def __call__(self, sparse_tensor):
        """
        Args:
            sparse_tensor: a torch sparse tensor in COO format with shape (B, C, W, H)
        Returns:
            The result of the convolution as a sparse tensor in COO format
        """
        coo = sparse_tensor.coalesce()
        values = coo.values()
        indices = coo.indices()
        
        B, C, W, H = sparse_tensor.shape
        out_height = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        output_values = []
        output_indices = []

        # Iterate over the non-zero elements in the sparse tensor
        for i in range(values.shape[0]):
            b, c, w, h = indices[:, i].tolist()
            value = values[i].item()
            for k in range(self.weights.shape[0]):
                for kh in range(self.kernel_size):
                    for kw in range(self.kernel_size):
                        h_out = (h - kw + self.padding) // self.stride
                        w_out = (w - kh + self.padding) // self.stride
                        if 0 <= h_out < out_height and 0 <= w_out < out_width:
                            output_values.append(value * self.weights[k, c, kh, kw].item())
                            output_indices.append([b, k, h_out, w_out])

        if self.bias is not None:
            for b in range(B):
                for k in range(self.weights.shape[0]):
                    for h_out in range(out_height):
                        for w_out in range(out_width):
                            output_values.append(self.bias[k].item())
                            output_indices.append([b, k, h_out, w_out])

        output_values = torch.tensor(output_values)
        output_indices = torch.tensor(output_indices).T
        
        size = (B, self.weights.shape[0], out_height, out_width)
        sparse_output = torch.sparse_coo_tensor(output_indices, output_values, size)
        
        return sparse_output

def conv2d_to_sparseconv2d(conv2d):
    """
    Transforms a torch.nn.Conv2d instance to a SparseConv2D instance
    
    Args:
        conv2d: instance of torch.nn.Conv2d
    
    Returns:
        instance of SparseConv2D
    """
    in_channels = conv2d.in_channels
    out_channels = conv2d.out_channels
    kernel_size = conv2d.kernel_size[0]
    stride = conv2d.stride[0]
    padding = conv2d.padding[0]
    weights = conv2d.weight.detach().numpy()
    if conv2d.bias is not None:
        bias = conv2d.bias.detach().numpy()
    else:
        bias = None
    
    return SparseConv2D(in_channels, out_channels, kernel_size, stride, padding, bias=(bias is not None), weights=weights, bias_val=bias)