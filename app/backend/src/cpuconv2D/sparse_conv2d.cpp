#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <vector>
#include <omp.h>

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

    at::Tensor operator()(const at::Tensor& sparse_tensor) {
        auto coo = sparse_tensor.coalesce();
        auto values = coo.values();
        auto indices = coo.indices();

        int B = sparse_tensor.size(0);
        int C = sparse_tensor.size(1);
        int W = sparse_tensor.size(2);
        int H = sparse_tensor.size(3);
        int out_height = (W + 2 * padding_ - kernel_size_) / stride_ + 1;
        int out_width = (H + 2 * padding_ - kernel_size_) / stride_ + 1;

        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        auto output = torch::zeros({B, out_channels_, out_height, out_width}, options);

        int nnz = values.size(0);

        #pragma omp parallel for
        for (int i = 0; i < nnz; ++i) {
            int b = indices[0][i].item<int>();
            int c = indices[1][i].item<int>();
            int w = indices[2][i].item<int>();
            int h = indices[3][i].item<int>();
            float value = values[i].item<float>();

            int group = c / group_in_channels_;
            int c_in_group = c % group_in_channels_;
            for (int k = 0; k < group_out_channels_; ++k) {
                int k_global = group * group_out_channels_ + k;
                for (int kh = 0; kh < kernel_size_; ++kh) {
                    for (int kw = 0; kw < kernel_size_; ++kw) {
                        int h_out = (h - kw + padding_) / stride_;
                        int w_out = (w - kh + padding_) / stride_;
                        if (h_out >= 0 && h_out < out_height && w_out >= 0 && w_out < out_width) {
                            #pragma omp atomic
                            output[b][k_global][h_out][w_out] += value * weights_[(k_global * group_in_channels_ * kernel_size_ * kernel_size_) + 
                                                                                  (c_in_group * kernel_size_ * kernel_size_) + 
                                                                                  (kh * kernel_size_) + kw];
                        }
                    }
                }
            }
        }

        if (!bias_.empty()) {
            #pragma omp parallel for
            for (int b = 0; b < B; ++b) {
                for (int k = 0; k < out_channels_; ++k) {
                    for (int h_out = 0; h_out < out_height; ++h_out) {
                        for (int w_out = 0; w_out < out_width; ++w_out) {
                            output[b][k][h_out][w_out] += bias_[k];
                        }
                    }
                }
            }
        }

        return output;
    }

private:
    int in_channels_, out_channels_, kernel_size_, stride_, padding_, groups_;
    int group_in_channels_, group_out_channels_;
    std::vector<float> weights_, bias_;
};


PYBIND11_MODULE(sparse_conv2d, m) {
    py::class_<SparseConv2D>(m, "SparseConv2D")
        .def(py::init<int, int, int, int, int, int, std::vector<float>, std::vector<float>>())
        .def("__call__", &SparseConv2D::operator());
}
