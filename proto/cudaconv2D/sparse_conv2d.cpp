#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <vector>
#include <unordered_map>

namespace py = pybind11;

class SparseConv2D {
public:
    SparseConv2D(int in_channels, int out_channels, int kernel_size, int stride, int padding, int groups,
                 std::vector<float> weights, std::vector<float> bias)
        : in_channels_(in_channels), out_channels_(out_channels), kernel_size_(kernel_size),
          stride_(stride), padding_(padding), groups_(groups), weights_(weights), bias_(bias) {
        group_in_channels_ = in_channels_ / groups_;
        group_out_channels_ = out_channels_ / groups_;
    }

    at::Tensor operator()(const at::Tensor& sparse_tensor, const at::Tensor& mask) {
        auto coo = sparse_tensor.coalesce();
        auto values = coo.values().to(at::kCUDA);
        auto indices = coo.indices().to(at::kCUDA);

        int B = sparse_tensor.size(0);
        int W = sparse_tensor.size(2);
        int H = sparse_tensor.size(3);
        int out_height = (W + 2 * padding_ - kernel_size_) / stride_ + 1;
        int out_width = (H + 2 * padding_ - kernel_size_) / stride_ + 1;

        auto options_values = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
        auto options_indices = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);

        // Allocate output tensors
        auto output_values = torch::zeros({B, out_channels_, out_height, out_width}, options_values);
        auto output_indices = torch::empty({4, values.size(0)}, options_indices);

        // Copy weights and bias to GPU
        auto weights_cuda = torch::tensor(weights_, options_values);
        auto bias_cuda = torch::tensor(bias_, options_values);

        int nnz = values.size(0);

        // Apply convolution in CUDA kernel
        sparse_conv2d_cuda(
            B, W, H, out_height, out_width, nnz,
            indices.data_ptr<int64_t>(),
            values.data_ptr<float>(),
            mask.to(at::kCUDA).data_ptr<float>(),
            weights_cuda.data_ptr<float>(),
            bias_cuda.data_ptr<float>(),
            group_in_channels_, group_out_channels_, kernel_size_,
            stride_, padding_, output_values.data_ptr<float>()
        );

        auto sparse_output = torch::sparse_coo_tensor(output_indices, output_values, {B, out_channels_, out_height, out_width}).coalesce();

        return sparse_output;
    }

private:
    int in_channels_, out_channels_, kernel_size_, stride_, padding_, groups_;
    int group_in_channels_, group_out_channels_;
    std::vector<float> weights_, bias_;
};

// CUDA function declaration
void sparse_conv2d_cuda(
    int B, int W, int H, int out_height, int out_width, int nnz,
    const int64_t* indices, const float* values, const float* mask,
    const float* weights, const float* bias, int group_in_channels,
    int group_out_channels, int kernel_size, int stride, int padding, float* output_values);

PYBIND11_MODULE(sparse_conv2d, m) {
    py::class_<SparseConv2D>(m, "SparseConv2D")
        .def(py::init<int, int, int, int, int, int, std::vector<float>, std::vector<float>>())
        .def("__call__", &SparseConv2D::operator());
}
