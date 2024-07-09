#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <vector>
#include <thread>
#include <mutex>

namespace py = pybind11;

class SparseConv2D {
public:
    SparseConv2D(int64_t in_channels, int64_t out_channels, int64_t kernel_size, int64_t stride, int64_t padding, 
                 std::vector<float> weights, std::vector<float> bias = {})
        : in_channels_(in_channels), out_channels_(out_channels), kernel_size_(kernel_size),
          stride_(stride), padding_(padding), weights_(weights), bias_(bias) {}

    at::Tensor operator()(const at::Tensor& sparse_tensor) {
        auto coo = sparse_tensor.coalesce();
        auto values = coo.values();
        auto indices = coo.indices();

        int64_t B = sparse_tensor.size(0);
        int64_t W = sparse_tensor.size(2);
        int64_t H = sparse_tensor.size(3);
        int64_t out_height = (W + 2 * padding_ - kernel_size_) / stride_ + 1;
        int64_t out_width = (H + 2 * padding_ - kernel_size_) / stride_ + 1;

        std::vector<float> output_values;
        std::vector<int64_t> output_indices;

        int64_t nnz = values.size(0);
        int num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads(num_threads);
        std::mutex output_mutex;

        auto worker = [&](int64_t start, int64_t end) {
            for (int64_t i = start; i < end; ++i) {
                int64_t b = indices[0][i].item<int64_t>();
                int64_t c = indices[1][i].item<int64_t>();
                int64_t w = indices[2][i].item<int64_t>();
                int64_t h = indices[3][i].item<int64_t>();
                float value = values[i].item<float>();

                for (int64_t k = 0; k < out_channels_; ++k) {
                    for (int64_t kh = 0; kh < kernel_size_; ++kh) {
                        for (int64_t kw = 0; kw < kernel_size_; ++kw) {
                            int64_t h_out = (h - kw + padding_) / stride_;
                            int64_t w_out = (w - kh + padding_) / stride_;
                            if (h_out >= 0 && h_out < out_height && w_out >= 0 && w_out < out_width) {
                                std::lock_guard<std::mutex> guard(output_mutex);
                                output_values.push_back(value * weights_[(k * in_channels_ * kernel_size_ * kernel_size_) + 
                                                                         (c * kernel_size_ * kernel_size_) + 
                                                                         (kh * kernel_size_) + kw]);
                                output_indices.push_back(b);
                                output_indices.push_back(k);
                                output_indices.push_back(h_out);
                                output_indices.push_back(w_out);
                            }
                        }
                    }
                }
            }
        };

        int64_t chunk_size = nnz / num_threads;
        for (int t = 0; t < num_threads; ++t) {
            int64_t start = t * chunk_size;
            int64_t end = (t == num_threads - 1) ? nnz : (t + 1) * chunk_size;
            threads[t] = std::thread(worker, start, end);
        }

        for (auto& t : threads) {
            t.join();
        }

        if (!bias_.empty()) {
            for (int64_t b = 0; b < B; ++b) {
                for (int64_t k = 0; k < out_channels_; ++k) {
                    for (int64_t h_out = 0; h_out < out_height; ++h_out) {
                        for (int64_t w_out = 0; w_out < out_width; ++w_out) {
                            output_values.push_back(bias_[k]);
                            output_indices.push_back(b);
                            output_indices.push_back(k);
                            output_indices.push_back(h_out);
                            output_indices.push_back(w_out);
                        }
                    }
                }
            }
        }

        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        auto output_values_tensor = torch::from_blob(output_values.data(), {static_cast<int64_t>(output_values.size())}, options).clone();
        auto output_indices_tensor = torch::from_blob(output_indices.data(), {4, static_cast<int64_t>(output_indices.size() / 4)}, torch::kInt64).clone();

        auto size = torch::IntArrayRef({B, out_channels_, out_height, out_width});
        return torch::sparse_coo_tensor(output_indices_tensor, output_values_tensor, size);
    }

private:
    int64_t in_channels_, out_channels_, kernel_size_, stride_, padding_;
    std::vector<float> weights_, bias_;
};

// Wrapper pour exposer à Python
PYBIND11_MODULE(sparse_conv2d, m) {
    py::class_<SparseConv2D>(m, "SparseConv2D")
        .def(py::init<int64_t, int64_t, int64_t, int64_t, int64_t, std::vector<float>, std::vector<float>>(), py::arg("in_channels"), py::arg("out_channels"), py::arg("kernel_size"), py::arg("stride"), py::arg("padding"), py::arg("weights"), py::arg("bias") = std::vector<float>())
        .def("__call__", &SparseConv2D::operator());
}
