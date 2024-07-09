import torch
import sparse_conv2d

# Dummy weights and bias for testing
weights = torch.randn(3, 3, 3, 3).numpy().flatten().tolist()
bias = torch.randn(3).numpy().flatten().tolist()

conv = sparse_conv2d.SparseConv2D(3, 3, 3, 1, 1, weights, bias)
indices = torch.tensor([[0, 0, 0, 1], [0, 1, 1, 1], [1, 0, 0, 0]], dtype=torch.int64).t()
values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
size = torch.Size([2, 3, 4, 4])
sparse_tensor = torch.sparse_coo_tensor(indices, values, size)

output = conv(sparse_tensor)
print(output)
