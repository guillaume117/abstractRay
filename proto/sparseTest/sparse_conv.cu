#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Noyau CUDA pour la convolution
__global__ void sparse_conv_forward_kernel(
    const float* input,
    const float* weights,
    float* output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_height,
    const int input_width,
    const int kernel_height,
    const int kernel_width,
    const int stride_height,
    const int stride_width,
    const int padding_height,
    const int padding_width,
    const int output_height,
    const int output_width
) {
    int batch_idx = blockIdx.x;
    int out_channel = blockIdx.y;
    int output_y = threadIdx.y;
    int output_x = threadIdx.x;

    if (batch_idx < batch_size && out_channel < out_channels &&
        output_y < output_height && output_x < output_width) {

        float result = 0.0;

        // Parcours des dimensions du noyau
        for (int in_channel = 0; in_channel < in_channels; ++in_channel) {
            for (int kh = 0; kh < kernel_height; ++kh) {
                for (int kw = 0; kw < kernel_width; ++kw) {
                    // Indices d'entrée
                    int input_y = output_y * stride_height + kh - padding_height;
                    int input_x = output_x * stride_width + kw - padding_width;

                    if (input_y >= 0 && input_y < input_height &&
                        input_x >= 0 && input_x < input_width) {
                        // Calcul des indices aplatis
                        int input_idx = batch_idx * in_channels * input_height * input_width +
                                        in_channel * input_height * input_width +
                                        input_y * input_width + input_x;
                        int weight_idx = out_channel * in_channels * kernel_height * kernel_width +
                                         in_channel * kernel_height * kernel_width +
                                         kh * kernel_width + kw;

                        result += input[input_idx] * weights[weight_idx];
                    }
                }
            }
        }

        // Écriture du résultat dans la sortie
        int output_idx = batch_idx * out_channels * output_height * output_width +
                         out_channel * output_height * output_width +
                         output_y * output_width + output_x;
        output[output_idx] = result;
    }
}

// Fonction appelée depuis le fichier C++
torch::Tensor sparse_conv_cuda_forward(
    torch::Tensor input,
    torch::Tensor weights,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding
) {
    // Extraction des dimensions
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);

    const int out_channels = weights.size(0);
    const int kernel_height = weights.size(2);
    const int kernel_width = weights.size(3);

    const int stride_height = stride[0];
    const int stride_width = stride[1];
    const int padding_height = padding[0];
    const int padding_width = padding[1];

    const int output_height = (input_height + 2 * padding_height - kernel_height) / stride_height + 1;
    const int output_width = (input_width + 2 * padding_width - kernel_width) / stride_width + 1;

    // Allocation du tenseur de sortie
    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, input.options());

    // Lancement du noyau CUDA
    dim3 blocks(batch_size, out_channels);
    dim3 threads(output_height, output_width);

    sparse_conv_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weights.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        stride_height,
        stride_width,
        padding_height,
        padding_width,
        output_height,
        output_width
    );

    return output;
}
