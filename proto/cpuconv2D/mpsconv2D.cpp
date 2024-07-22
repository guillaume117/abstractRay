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
        auto values = coo.values();
        auto indices = coo.indices();

        int B = sparse_tensor.size(0);
        int W = sparse_tensor.size(2);
        int H = sparse_tensor.size(3);
        int out_height = (W + 2 * padding_ - kernel_size_) / stride_ + 1;
        int out_width = (H + 2 * padding_ - kernel_size_) / stride_ + 1;

        std::unordered_map<std::vector<int64_t>, float, torch::hash<std::vector<int64_t>>> out_map;

        int nnz = values.size(0);

        at::Tensor output = torch::zeros({B, out_channels_, out_height, out_width}, torch::TensorOptions().device(torch::kMPS));

        auto weights = torch::from_blob(weights_.data(), {out_channels_, in_channels_ / groups_, kernel_size_, kernel_size_}).to(torch::kMPS);
        auto bias = torch::from_blob(bias_.data(), {out_channels_}).to(torch::kMPS);

        // Convert indices and values to MPS
        auto indices_mps = indices.to(torch::kMPS);
        auto values_mps = values.to(torch::kMPS);
        auto mask_mps = mask.to(torch::kMPS);

        // Ensure all data is on the same device
        weights = weights.to(torch::kMPS);
        bias = bias.to(torch::kMPS);

        // Parallel processing on GPU using at::parallel_for
        at::parallel_for(0, nnz, 0, [&](int64_t start, int64_t end) {
            for (int64_t i = start; i < end; ++i) {
                int b = indices_mps[0][i].item<int>();
                int c = indices_mps[1][i].item<int>();
                int w = indices_mps[2][i].item<int>();
                int h = indices_mps[3][i].item<int>();
                float value = values_mps[i].item<float>();

                // Apply mask
                float mask_value = mask_mps[c][w][h].item<float>();
                value *= mask_value;

                int group = c / group_in_channels_;
                int c_in_group = c % group_in_channels_;
                for (int k = 0; k < group_out_channels_; ++k) {
                    int k_global = group * group_out_channels_ + k;
                    for (int kh = 0; kh < kernel_size_; ++kh) {
                        for (int kw = 0; kw < kernel_size_; ++kw) {
                            int h_out = (h - kw + padding_) / stride_;
                            int w_out = (w - kh + padding_) / stride_;

                            if (h_out >= 0 && h_out < out_height && w_out >= 0 && w_out < out_width) {
                                std::vector<int64_t> key = {b, k_global, h_out, w_out};
                                out_map[key] += value * weights[(k_global * group_in_channels_ * kernel_size_ * kernel_size_) + 
                                                                (c_in_group * kernel_size_ * kernel_size_) + 
                                                                (kh * kernel_size_) + kw].item<float>();
                            }
                        }
                    }
                }
            }
        });

        if (!bias_.empty()) {
            at::parallel_for(0, B, 0, [&](int64_t start, int64_t end) {
                for (int64_t b = start; b < end; ++b) {
                    for (int k = 0; k < out_channels_; ++k) {
                        for (int h_out = 0; h_out < out_height; ++h_out) {
                            for (int w_out = 0; w_out < out_width; ++w_out) {
                                std::vector<int64_t> key = {b, k, h_out, w_out};
                                out_map[key] += bias_[k];
                            }
                        }
                    }
                }
            });
        }

        std::vector<int64_t> out_indices_vec;
        std::vector<float> out_values_vec;

        for (const auto& pair : out_map) {
            const auto& key = pair.first;
            float value = pair.second;
            out_indices_vec.insert(out_indices_vec.end(), key.begin(), key.end());
            out_values_vec.push_back(value);
        }

        auto options_values = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kMPS);
        auto options_indices = torch::TensorOptions().dtype(torch::kInt64).device(torch::kMPS);
        auto output_indices = torch::tensor(out_indices_vec, options_indices).reshape({-1, 4}).transpose(0, 1);
        auto output_values = torch::tensor(out_values_vec, options_values);
        auto sparse_output = torch::sparse_coo_tensor(output_indices, output_values, {B, out_channels_, out_height, out_width}).coalesce();

        return sparse_output;
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
