import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        # Initialize the weights and bias for reproducibility
        #self.conv.weight.data = torch.tensor([[[[1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, -1.0]]]])
        self.conv.bias.zero_()

    def forward(self, x):
        return self.conv(x)

def determine_matrix(function_epsilon, m, p):
    """
    Determine the representation matrix of a linear function f: R^m -> R^p.

    Args:
        function_epsilon (function): The function representing the layer.
        m (torch.Tensor): The input tensor.
        p (torch.Tensor): The output tensor.

    Returns:
        torch.Tensor: The representation matrix of the function.
    """
    m_in = m.numel()
    p_out = p.numel()
    model = lambda x: function_epsilon(x)
    
    # Indices and values for the sparse matrix
    indices = []
    values = []

    for i in range(m_in):
        # Create base vector dynamically
        base_vector = torch.zeros(m_in)
        base_vector[i] = 1.0
        
        result = model(base_vector.unsqueeze(0).view_as(torch.zeros(m))).flatten()
        non_zero_indices = torch.nonzero(result, as_tuple=True)[0]
        non_zero_values = result[non_zero_indices]
        
        if non_zero_values.numel() > 0:
            indices.append(torch.stack([non_zero_indices, torch.full_like(non_zero_indices, i)], dim=0))
            values.append(non_zero_values)

    if indices:
        indices = torch.cat(indices, dim=1)
        values = torch.cat(values)
    else:
        indices = torch.empty((2, 0), dtype=torch.long)
        values = torch.empty(0, dtype=m.dtype)
    
    A_sparse = torch.sparse_coo_tensor(indices, values, (p_out, m_in)).coalesce()
    
    return A_sparse

with torch.no_grad():

    # Initialiser le modèle
    model = Model()

    # Définir la fonction epsilon
    def function_epsilon(x):
        return model(x)

    # Créer des tensors d'entrée et de sortie
    input_tensor = torch.randn(1,3,32,32)
    input_size =input_tensor.size()
    output_size = model(input_tensor).size()

    output_tensor = function_epsilon(input_tensor)


    A_sparse = determine_matrix(function_epsilon, input_size, output_size)
    print(A_sparse.to_dense())
    # Vérifier que A @ x.flatten() = function(x).flatten()
    x_flat = input_tensor.flatten()
    Ax_flat = torch.sparse.mm(A_sparse, x_flat.unsqueeze(1)).squeeze()
    function_x_flat = function_epsilon(input_tensor).flatten()


    print("Input Tensor Flattened:", x_flat)
    print("A @ x.flatten():", Ax_flat)
    print("function(x).flatten():", function_x_flat)
    print("Différence entre A @ x.flatten() et function(x).flatten():", torch.norm(Ax_flat - function_x_flat).item())
