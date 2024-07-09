import torch
import sparse_conv2d

# Dummy weights and bias for testing
weights_tensor = torch.randn(3, 1, 3, 3)
bias_tensor = torch.zeros(3)
weights = weights_tensor.numpy().flatten().tolist()
bias = bias_tensor.numpy().flatten().tolist()
bias = []

conv = sparse_conv2d.SparseConv2D(3, 3, 3, 1, 1,3, weights,bias)
indices = torch.tensor([[0, 0, 0, 1], [0, 1, 1, 1], [1, 0, 0, 0]], dtype=torch.int64).t()
values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
tensor_test = torch.randn(200000,3,42,42)
tensor_test = torch.where(tensor_test>4,tensor_test,0).to_sparse()
print(f'tensor_test nnz = {tensor_test._nnz()}')
size = torch.Size([20000, 3, 4, 4])
sparse_tensor = torch.sparse_coo_tensor(indices, values, size)

output = conv(tensor_test,torch.ones(3,42,42))
conv2 = torch.nn.Conv2d(3,3,3,1,1,groups=3)
conv2.weight.data = weights_tensor
conv2.bias.data = bias_tensor
with torch.no_grad():
    print(torch.sum(output-conv2(tensor_test.to_dense())))
