import torch
import custom_sparse

coo_tensor = torch.randn(80000,1,1,2).to_sparse_coo()
print(coo_tensor)


print([*coo_tensor.size()])
custom = custom_sparse.coo_to_custom(coo_tensor.indices(), coo_tensor.values(),[*coo_tensor.size()])

print(custom.data)
# Conversion de custom format vers COO tensor
indices, values = custom_sparse.custom_to_coo(custom)
print(custom.size)


print(indices)
coo_tensor_back = torch.sparse_coo_tensor(indices, values, custom.size).coalesce()
print(torch.sum(coo_tensor_back)-torch.sum(coo_tensor))


