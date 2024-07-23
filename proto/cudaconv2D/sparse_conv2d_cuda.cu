#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void sparse_conv2d_kernel(
    int B, int W, int H, int out_height, int out_width, int nnz,
    const int64_t* indices, const float* values, const float* mask,
    const float* weights, const float* bias, int group_in_channels,
    int group_out_channels, int kernel_size, int stride, int padding, float* output_values) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnz) return;

    int b = indices[i * 4 + 0];
    int c = indices[i * 4 + 1];
    int w = indices[i * 4 + 2];
    int h = indices[i * 4 + 3];
    float value = values[i];

    // Apply mask
    float mask_value = mask[c * W * H + w * H + h];
    value *= mask_value;

    int group = c / group_in_channels;
    int c_in_group = c % group_in_channels;
    for (int k = 0; k < group_out_channels; ++k) {
        int k_global = group * group_out_channels + k;
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int h_out = (h - kw + padding) / stride;
                int w_out = (w - kh + padding) / stride;

                if (h_out >= 0 && h_out < out_height && w_out >= 0 && w_out < out_width) {
                    atomicAdd(&output_values[((b * out_channels + k_global) * out_height + h_out) * out_width + w_out],
                              value * weights[(k_global * group_in_channels * kernel_size * kernel_size) +
                                              (c_in_group * kernel_size * kernel_size) +
                                              (kh * kernel_size) + kw]);
                }
            }
        }
    }

    if (bias != nullptr) {
        for (int b = 0; b < B; ++b) {
            for (int k = 0; k < out_channels; ++k) {
                for (int h_out = 0; h_out < out_height; ++h_out) {
                    for (int w_out = 0; w_out < out_width; ++w_out) {
                        atomicAdd(&output_values[((b * out_channels + k) * out_height + h_out) * out_width + w_out], bias[k]);
                    }
                }
            }
        }
    }
}

void sparse_conv2d_cuda(
    int B, int W, int H, int out_height, int out_width, int nnz,
    const int64_t* indices, const float* values, const float* mask,
    const float* weights, const float* bias, int group_in_channels,
    int group_out_channels, int kernel_size, int stride, int padding, float* output_values) {

    int threads = 1024;
    int blocks = (nnz + threads - 1) / threads;

    sparse_conv2d_kernel<<<blocks, threads>>>(
        B, W, H, out_height, out_width, nnz, indices, values, mask,
        weights, bias, group_in_channels, group_out_channels, kernel_size,
        stride, padding, output_values);

    cudaDeviceSynchronize();
}
