import torch
import sparse_conv2d
import json
import os

output_dir = "test_results"
os.makedirs(output_dir, exist_ok=True)


batch_sizes = [10,100]
input_channels = [1, 32]
output_channels = [16, 32]
kernel_sizes = [1, 2, 3]
stride_values = [1, 2]
padding_values = [0, 1]



test_results = []


for batch_size in batch_sizes:
    for in_channels in input_channels:
        for out_channels in output_channels:

            for kernel_size in kernel_sizes:
                weights_tensor = torch.randn((out_channels, in_channels, kernel_size, kernel_size), dtype=torch.float32)
                bias_tensor = torch.zeros(out_channels, dtype=torch.float32)
                for stride in stride_values:
                    for padding in padding_values:
                        print(f'Test with batch_size={batch_size}, in_channels={in_channels}, out_channels={out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding}')

                        conv2 = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
                        conv = sparse_conv2d.SparseConv2D(in_channels, out_channels, kernel_size, stride, padding, 1, weights_tensor.flatten().tolist(), [])

       
                        tensor_test = torch.randn(batch_size, in_channels, 42, 42)
                        tensor_test = torch.where(torch.abs(tensor_test) < 0.01, tensor_test, torch.tensor(0.0)).to_sparse()

                        print(f'tensor_test nnz = {tensor_test._nnz()}')
                        size = torch.Size([batch_size, in_channels, 42, 42])
                        sparse_tensor = torch.sparse_coo_tensor(tensor_test.indices(), tensor_test.values(), size, dtype=torch.float32)

                        output = conv(tensor_test, torch.ones(in_channels, 42, 42))
                   

                        conv2.weight.data = weights_tensor
                        conv2.bias.data = bias_tensor
                        result_torch = conv2(tensor_test.to_dense())

                        nnz_difference = output._nnz() != result_torch.to_sparse()._nnz()

                       
                        test_result = {
                            'batch_size': batch_size,
                            'in_channels': in_channels,
                            'out_channels': out_channels,
                            'kernel_size': kernel_size,
                            'stride': stride,
                            'padding': padding,
                            'output_sum_diff': torch.sum(output.to_dense() - result_torch).item(),
                            'nnz_cpp_result': output._nnz(),
                            'nnz_torch_eval': result_torch.to_sparse()._nnz(),
                            'nnz_difference': nnz_difference
                        }
                        print(test_result)

                        test_results.append(test_result)

with open(os.path.join(output_dir, 'test_results_cpp2.json'), 'w') as f:
    json.dump(test_results, f, indent=4)

print("Tests completed and results saved.")
