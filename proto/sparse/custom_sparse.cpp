#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <unordered_map>
#include <vector>
#include <tuple>
#include <functional>
#include <tuple>

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


struct CustomSparse {
    std::unordered_map<std::tuple<int8_t, int8_t, int8_t>, std::vector<std::pair<int32_t, float>>> data;
    std::vector<int64_t> size;
    int64_t nnz;
};
CustomSparse coo_to_custom(torch::Tensor indices, torch::Tensor values, std::vector<int64_t> size) {
    CustomSparse custom;
    custom.size = size;
    custom.nnz = values.size(0);

    auto indices_accessor = indices.accessor<int64_t, 2>();
    auto values_accessor = values.accessor<float, 1>();

    for (int64_t i = 0; i < custom.nnz; ++i) {
        int32_t b_index = static_cast<int32_t>(indices_accessor[0][i]);
        int8_t c_index = static_cast<int8_t>(indices_accessor[1][i]);
        int8_t h_index = static_cast<int8_t>(indices_accessor[2][i]);
        int8_t w_index = static_cast<int8_t>(indices_accessor[3][i]);
        float value = values_accessor[i];

        std::tuple<int8_t, int8_t, int8_t> key = std::make_tuple(c_index, h_index, w_index);
        custom.data[key].emplace_back(b_index, value);
    }

    return custom;
}

std::pair<torch::Tensor, torch::Tensor> custom_to_coo(CustomSparse custom) {
    int64_t nnz = custom.nnz;
    torch::Tensor indices = torch::empty({4, nnz}, torch::kInt64);
    torch::Tensor values = torch::empty(nnz, torch::kFloat32);

    auto indices_accessor = indices.accessor<int64_t, 2>();
    auto values_accessor = values.accessor<float, 1>();

    int64_t i = 0;
    for (const auto& entry : custom.data) {
        const auto& key = entry.first;
        const auto& pairs = entry.second;

        int8_t c_index, h_index, w_index;
        std::tie(c_index, h_index, w_index) = key;

        for (const auto& pair : pairs) {
            indices_accessor[0][i] = pair.first;
            indices_accessor[1][i] = c_index;
            indices_accessor[2][i] = h_index;
            indices_accessor[3][i] = w_index;
            values_accessor[i] = pair.second;
            ++i;
        }
    }

    return std::make_pair(indices, values);
}
PYBIND11_MODULE(custom_sparse, m) {
    pybind11::class_<CustomSparse>(m, "CustomSparse")
        .def(pybind11::init<>())
        .def_readwrite("data", &CustomSparse::data)
        .def_readwrite("size", &CustomSparse::size)
        .def_readwrite("nnz", &CustomSparse::nnz);

    m.def("coo_to_custom", &coo_to_custom, "Convert COO tensor to custom sparse format",
          pybind11::arg("indices"), pybind11::arg("values"), pybind11::arg("size"));
    m.def("custom_to_coo", &custom_to_coo, "Convert custom sparse format to COO tensor");
}