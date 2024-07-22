#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <vector>
#include <unordered_map>
#include <omp.h>
#include <mutex>

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

        using key_t = std::vector<int64_t>;
        struct ValueBatch {
            std::vector<float> values;
            std::vector<int> batch_indices;
        };
        std::unordered_map<key_t, ValueBatch, torch::hash<key_t>> index_map;
        std::mutex index_map_mutex;

        int nnz = values.size(0);

        // Step 1: Create the collection of dictionaries
        #pragma omp parallel for
        for (int i = 0; i < nnz; ++i) {
            int b = indices[0][i].item<int>();
            int c = indices[1][i].item<int>();
            int w = indices[2][i].item<int>();
            int h = indices[3][i].item<int>();
            float value = values[i].item<float>();

            key_t key = {c, w, h};
            {
                std::lock_guard<std::mutex> guard(index_map_mutex);
                index_map[key].values.push_back(value);
                index_map[key].batch_indices.push_back(b);
            }
        }

        std::unordered_map<key_t, float, torch::hash<key_t>> out_map;
        std::mutex out_map_mutex;

        std::vector<key_t> keys;
        keys.reserve(index_map.size());
        for (const auto& pair : index_map) {
            keys.push_back(pair.first);
        }

        // Step 2: Apply the convolution operation
        #pragma omp parallel for
        for (size_t i = 0; i < keys.size(); ++i) {
            const auto& key = keys[i];
            const auto& value_batch = index_map[key];
            const auto& values = value_batch.values;
            const auto& batch_indices = value_batch.batch_indices;

            int c = key[0];
            int w = key[1];
            int h = key[2];

            int group = c / group_in_channels_;
            int c_in_group = c % group_in_channels_;
            for (int k = 0; k < group_out_channels_; ++k) {
                int k_global = group * group_out_channels_ + k;
                for (int kh = 0; kh < kernel_size_; ++kh) {
                    for (int kw = 0; kw < kernel_size_; ++kw) {
                        int h_out = (h - kw + padding_) / stride_;
                        int w_out = (w - kh + padding_) / stride_;
                        if (h_out >= 0 && h_out < out_height && w_out >= 0 && w_out < out_width) {
                            for (size_t i = 0; i < values.size(); ++i) {
                                int b = batch_indices[i];
                                float value = values[i] * mask[c][w][h].item<float>();

                                key_t out_key = {b, k_global, h_out, w_out};
                                {
                                    std::lock_guard<std::mutex> guard(out_map_mutex);
                                    out_map[out_key] += value * weights_[(k_global * group_in_channels_ * kernel_size_ * kernel_size_) +
                                                                         (c_in_group * kernel_size_ * kernel_size_) +
                                                                         (kh * kernel_size_) + kw];
                                }
                            }
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
                            key_t key = {b, k, h_out, w_out};
                            {
                                std::lock_guard<std::mutex> guard(out_map_mutex);
                                out_map[key] += bias_[k];
                            }
                        }
                    }
                }
            }
        }

        std::vector<int64_t> out_indices;
        std::vector<float> out_values;

        for (const auto& pair : out_map) {
            const auto& key = pair.first;
            float value = pair.second;
            out_indices.insert(out_indices.end(), key.begin(), key.end());
            out_values.push_back(value);
        }

        auto options_values = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        auto options_indices = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
        auto output_indices = torch::tensor(out_indices, options_indices).reshape({-1, 4}).transpose(0, 1);
        auto output_values = torch::tensor(out_values, options_values);
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
