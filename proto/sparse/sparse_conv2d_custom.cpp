#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <vector>
#include <unordered_map>
#include <tuple>
#include <omp.h>
#include <mutex>

namespace py = pybind11;

// Définir la structure CustomSparse
struct CustomSparse {
    std::unordered_map<std::tuple<int8_t, int8_t, int8_t>, std::vector<std::pair<int32_t, float>>> data;
    std::vector<int64_t> size;
    int64_t nnz;
};

// Spécialisation de std::hash pour std::tuple<int8_t, int8_t, int8_t>
namespace std {
    template <>
    struct hash<std::tuple<int8_t, int8_t, int8_t>> {
        size_t operator()(const std::tuple<int8_t, int8_t, int8_t>& key) const {
            auto hash1 = std::hash<int8_t>{}(std::get<0>(key));
            auto hash2 = std::hash<int8_t>{}(std::get<1>(key));
            auto hash3 = std::hash<int8_t>{}(std::get<2>(key));
            return hash1 ^ (hash2 << 1) ^ (hash3 << 2); // Combine hashes
        }
    };

    template <>
    struct equal_to<std::tuple<int8_t, int8_t, int8_t>> {
        bool operator()(const std::tuple<int8_t, int8_t, int8_t>& lhs, const std::tuple<int8_t, int8_t, int8_t>& rhs) const {
            return std::get<0>(lhs) == std::get<0>(rhs) &&
                   std::get<1>(lhs) == std::get<1>(rhs) &&
                   std::get<2>(lhs) == std::get<2>(rhs);
        }
    };
}
// Définir la classe SparseConv2D
class SparseConv2D {
public:
    SparseConv2D(int in_channels, int out_channels, int kernel_size, int stride, int padding, int groups,
                 std::vector<float> weights, std::vector<float> bias)
        : in_channels_(in_channels), out_channels_(out_channels), kernel_size_(kernel_size),
          stride_(stride), padding_(padding), groups_(groups), weights_(weights), bias_(bias) {
        group_in_channels_ = in_channels_ / groups_;
        group_out_channels_ = out_channels_ / groups_;
    }

    CustomSparse operator()(const CustomSparse& custom_sparse, const at::Tensor& mask) {
        int B = custom_sparse.size[0];
        int W = custom_sparse.size[2];
        int H = custom_sparse.size[3];
        int out_height = (W + 2 * padding_ - kernel_size_) / stride_ + 1;
        int out_width = (H + 2 * padding_ - kernel_size_) / stride_ + 1;

        using key_t = std::tuple<int32_t, int8_t, int8_t, int8_t>;
        std::unordered_map<key_t, float> out_map;
        std::mutex out_map_mutex;

        // Apply the convolution operation
        #pragma omp parallel for
        for (const auto& entry : custom_sparse.data) {
            const auto& key = entry.first;
            const auto& value_batch = entry.second;
            const auto& values = value_batch;

            int c = std::get<0>(key);
            int w = std::get<1>(key);
            int h = std::get<2>(key);

            int group = c / group_in_channels_;
            int c_in_group = c % group_in_channels_;
            for (int k = 0; k < group_out_channels_; ++k) {
                int k_global = group * group_out_channels_ + k;
                for (int kh = 0; kh < kernel_size_; ++kh) {
                    for (int kw = 0; kw < kernel_size_; ++kw) {
                        int h_out = (h - kh + padding_) / stride_;
                        int w_out = (w - kw + padding_) / stride_;
                        if (h_out >= 0 && h_out < out_height && w_out >= 0 && w_out < out_width) {
                            for (const auto& pair : values) {
                                int b = pair.first;
                                float value = pair.second * mask[c][w][h].item<float>();

                                key_t out_key = std::make_tuple(b, k_global, h_out, w_out);
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
                            key_t key = std::make_tuple(b, k, h_out, w_out);
                            {
                                std::lock_guard<std::mutex> guard(out_map_mutex);
                                out_map[key] += bias_[k];
                            }
                        }
                    }
                }
            }
        }

        CustomSparse custom_output;
        custom_output.size = {B, out_channels_, out_height, out_width};
        custom_output.nnz = out_map.size();

        for (const auto& pair : out_map) {
            const auto& key = pair.first;
            float value = pair.second;

            int b = std::get<0>(key);
            int c = std::get<1>(key);
            int h = std::get<2>(key);
            int w = std::get<3>(key);

            std::tuple<int8_t, int8_t, int8_t> out_key = std::make_tuple(c, h, w);
            custom_output.data[out_key].emplace_back(b, value);
        }

        return custom_output;
    }

private:
    int in_channels_, out_channels_, kernel_size_, stride_, padding_, groups_;
    int group_in_channels_, group_out_channels_;
    std::vector<float> weights_, bias_;
};

PYBIND11_MODULE(custom_conv, m) {
    py::class_<SparseConv2D>(m, "SparseConv2D")
        .def(py::init<int, int, int, int, int, int, std::vector<float>, std::vector<float>>())
        .def("__call__", &SparseConv2D::operator());

    py::class_<CustomSparse>(m, "CustomSparse")
        .def(py::init<>())
        .def_readwrite("data", &CustomSparse::data)
        .def_readwrite("size", &CustomSparse::size)
        .def_readwrite("nnz", &CustomSparse::nnz);
}
