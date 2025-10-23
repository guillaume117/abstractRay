#include <torch/extension.h>
#include <vector>

// Déclaration de la fonction CUDA uniquement (pas de définition)
torch::Tensor sparse_conv_cuda_forward(
    torch::Tensor input,
    torch::Tensor weights,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding
);

// Définition de l'interface Python
torch::Tensor sparse_conv_forward(
    torch::Tensor input,
    torch::Tensor weights,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding
) {
    return sparse_conv_cuda_forward(input, weights, stride, padding);
}

// Enregistrement avec PyBind11
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sparse_conv_forward, "Sparse Convolution Forward (CUDA)");
}
