import torch
import torch.nn.functional as F
import ray



class SparseConv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, dilation=1, bias=True, weights=None, bias_val=None,num_workers = 1):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.dilation = dilation
        self.num_workers = num_workers

        # Initialize weights and biases
        if weights is None:
            self.weights = torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size)
        else:
            self.weights = torch.tensor(weights, dtype=torch.float32)

        if bias:
            if bias_val is None:
                self.bias = torch.randn(out_channels)
            else:
                self.bias = torch.tensor(bias_val, dtype=torch.float32)
        else:
            self.bias = None

    def __call__(self, sparse_tensor, mask):
        """
        Args:
            sparse_tensor: a torch sparse tensor in COO format with shape (B, C, H, W)
        Returns:
            The result of the convolution as a sparse tensor in COO format
        """
        coo = sparse_tensor.coalesce()
        values = coo.values()
        indices = coo.indices()

        B, C, H, W = sparse_tensor.shape
        out_height = (H + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        out_width = (W + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1

        output = {
            'values': [],
            'indices': []
        }

        # Adjust this according to the number of available cores

        # Split the indices and values among workers
        chunk_size = (values.shape[0] + self.num_workers - 1) // self.num_workers
        chunks = [(indices[:, i:i + chunk_size], values[i:i + chunk_size]) for i in range(0, values.shape[0], chunk_size)]

        # Parallel processing with Ray
        worker = SparseWorker.remote(self.weights, self.bias, self.kernel_size, self.stride, self.padding, self.dilation, self.groups)
        results = ray.get([worker.process_chunk.remote(chunk, mask, out_height, out_width, C) for chunk in chunks])

        for res in results:
            output['values'].extend(res['values'])
            output['indices'].extend(res['indices'])

        # Apply bias
        if self.bias is not None:
            for b in range(B):
                for k in range(self.weights.shape[0]):
                    for h_out in range(out_height):
                        for w_out in range(out_width):
                            output['values'].append(self.bias[k].item())
                            output['indices'].append([b, k, h_out, w_out])

        output['values'] = torch.tensor(output['values'])
        output['indices'] = torch.tensor(output['indices']).T

        size = (B, self.weights.shape[0], out_height, out_width)
        sparse_output = torch.sparse_coo_tensor(output['indices'], output['values'], size)

        return sparse_output, torch.sum(torch.abs(sparse_output), dim=0)

@ray.remote(num_cpus=14)
class SparseWorker:
    def __init__(self, weights, bias, kernel_size, stride, padding, dilation, groups):
        self.weights = weights
        self.bias = bias
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def process_chunk(self, chunk, mask, out_height, out_width, C):
        indices, values = chunk
        output = {
            'values': [],
            'indices': []
        }

        for i in range(values.shape[0]):
            b, c, h, w = indices[:, i].tolist()
            value = values[i].item()

            for kh in range(self.kernel_size):
                for kw in range(self.kernel_size):
                    h_out = (h + self.padding - kh * self.dilation) // self.stride
                    w_out = (w + self.padding - kw * self.dilation) // self.stride

                    if 0 <= h_out < out_height and 0 <= w_out < out_width:
                        for k in range(self.weights.shape[0]):
                            group_idx = c // (C // self.groups)
                            weight_idx = c % (C // self.groups)
                            conv_value = value * mask[0, c, h, w].item() * self.weights[k, weight_idx, kh, kw].item()

                            output['values'].append(conv_value)
                            output['indices'].append([b, k, h_out, w_out])

        return output

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
    groups = conv2d.groups
    weights = conv2d.weight.detach().numpy()
    dilation = conv2d.dilation[0]
    if conv2d.bias is not None:
        bias = conv2d.bias.detach().numpy()
    else:
        bias = None

    return SparseConv2D(in_channels, out_channels, kernel_size, stride, padding, groups=groups, dilation=dilation, bias=(bias is not None), weights=weights, bias_val=bias)
