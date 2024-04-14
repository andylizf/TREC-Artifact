#include <cuda.h>
#include <iostream>
#include <torch/extension.h>

std::vector<at::Tensor> conv_deep_reuse_forward(const at::Tensor& input,
    const at::Tensor& weights,
    const at::Tensor& bias,
    const at::Tensor& random_vectors,
    const int64_t pad_height,
    const int64_t pad_width,
    const int64_t stride_height,
    const int64_t stride_width,
    const int64_t param_L,
    const int64_t param_H,
    const bool do_bias,
    const bool is_training,
    const bool print_rc);

std::vector<at::Tensor> conv_deep_reuse_backward(
    // const at::IntArrayRef input_size,
    const at::Tensor input_row, // {n_matrices, num_row, param_L}
    const at::Tensor inputCentroids, // {n_matrices, max_buckets, param_L}
    const at::Tensor weights,
    const at::Tensor gradOutput, // {N, K, outH, outW}
    const at::Tensor vector_index, // {n_matrices, num_rows}
    const at::Tensor vector_ids, // {n_matrices, num_rows}
    const at::Tensor buckets_count, // {n_matrices, max_buckets}
    const at::Tensor buckets_index, // {n_matrices, total_buckets}
    const at::Tensor buckets_index_inv, // {n_matrices, max_buckets}
    const at::Tensor random_vectors, // {L, H}
    const int64_t input_height,
    const int64_t input_width,
    const int64_t pad_height,
    const int64_t pad_width,
    const int64_t stride_height,
    const int64_t stride_width,
    const int64_t param_H,
    const float alpha,
    const float sigma,
    const bool do_bias);

at::Tensor input, weights, bias, random_vectors, input_row, inputCentroids, gradOutput, vector_index, vector_ids, buckets_count, buckets_index, buckets_index_inv;
int64_t pad_height, pad_width, stride_height, stride_width;
bool do_bias, is_training, print_rc;
float alpha, sigma;

// void cifar_params()
// {
//     input = torch::rand({ 10, 16, 16, 16 }).to(torch::kCUDA);
//     weights = torch::rand({ 64, 16, 3, 3 }).to(torch::kCUDA);
//     bias = torch::rand({ 64 }).to(torch::kCUDA);
//     random_vectors = torch::rand({ 48, 5 }).to(torch::kCUDA);
//     pad_height = 1;
//     pad_width = 1;
//     stride_height = 1;
//     stride_width = 1;
//     param_L = 48;
//     param_H = 5;
//     do_bias = true;
//     is_training = true;
//     print_rc = false;

//     input_row = torch::rand({ 3, 2560, 48 }).to(torch::kCUDA);
//     inputCentroids = torch::rand({ 3, 32, 48 }).to(torch::kCUDA);
//     gradOutput = torch::rand({ 10, 64, 16, 16 }).to(torch::kCUDA);
//     vector_index = torch::rand({ 3, 2560 }).to(torch::kCUDA);

//     auto options = torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA);
//     vector_index = torch::tensor(
// #include "/home/lizhifei/TREC-Artifact/vector_index.txt"
//         , options);
//     vector_ids = torch::tensor(
// #include "/home/lizhifei/TREC-Artifact/vector_ids.txt"
//         , options);
//     buckets_count = torch::tensor({ { 86, 58, 257, 429, 1, 26, 7, 318, 307, 206, 261, 5, 1, 43, 27, 52, 23, 184, 91, 3, 21, 14, 69, 38, 1, 23, 9, 0, 0, 0, 0, 0 },
//                                       { 81, 99, 316, 486, 3, 5, 31, 12, 163, 79, 493, 219, 6, 6, 39, 25, 82, 27, 77, 71, 1, 1, 5, 2, 49, 28, 88, 41, 11, 2, 9, 3 },
//                                       { 70, 119, 177, 190, 8, 28, 30, 37, 117, 273, 417, 174, 33, 70, 69, 19, 123, 55, 249, 140, 8, 3, 1, 4, 33, 46, 31, 24, 2, 2, 6, 2 } },
//         options);
//     buckets_index = torch::tensor({ { 0, 1, 2, 3, 4, -1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, -1, -1, 19, -1, 20, 21, 22, 23, 24, -1, 25, 26 },
//                                       { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 },
//                                       { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 } },
//         options);
//     buckets_index_inv = torch::tensor({ { 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 24, 25, 26, 27, 28, 30, 31, 0, 0, 0, 0, 0 },
//                                           { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 },
//                                           { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 } },
//         options);
//     input_height = 16;
//     input_width = 16;
//     alpha = 10000;
//     sigma = 0.00625;
// }

constexpr int64_t batch_size = 100;
constexpr int64_t input_channels = 3;
constexpr int64_t input_height = 224, input_width = 224;
constexpr int64_t output_channels = 64;
constexpr int64_t kernel_width = 7, kernel_height = 7;
constexpr int64_t param_L = 49, param_H = 5;

void resnet_params()
{
    input = torch::rand({ batch_size, input_channels, input_height, input_width }).to(torch::kCUDA);
    weights = torch::rand({ output_channels, input_channels, kernel_height, kernel_width }).to(torch::kCUDA);
    bias = torch::rand({ output_channels }).to(torch::kCUDA);
    random_vectors = torch::rand({ param_L, param_H }).to(torch::kCUDA);
    pad_height = 3;
    pad_width = 3;
    stride_height = 2;
    stride_width = 2;
    do_bias = true;
    is_training = true;
    print_rc = false;
    alpha = 10000;
    sigma = 0.00625;
}

int main()
{
    if (!torch::cuda::is_available()) {
        std::cerr << "CUDA is not available. Please check your configuration." << std::endl;
        return 1;
    }
    std::cout << "This file is used to profile the performance of the ConvDeepReuseBackward and ConvDeepReuseForward." << std::endl;

    resnet_params();

    auto res = conv_deep_reuse_forward(
        input,
        weights,
        bias,
        random_vectors,
        pad_height,
        pad_width,
        stride_height,
        stride_width,
        param_L,
        param_H,
        do_bias,
        is_training,
        print_rc);

    std::cout << "Forward pass finished." << std::endl;

    int output_height = res[0].size(2);
    int output_width = res[0].size(3);
    auto restructed_ouput_shape = res[0].sizes();
    assert(restructed_ouput_shape[0] == 100);
    assert(restructed_ouput_shape[1] == 64);
    assert(restructed_ouput_shape[2] == output_height);
    assert(restructed_ouput_shape[3] == output_width);
    assert(output_height == (input_height + 2 * pad_height - 7) / stride_height + 1);
    assert(output_width == (input_width + 2 * pad_width - 7) / stride_width + 1);
    gradOutput = torch::rand({ 100, 64, output_height, output_width }).to(torch::kCUDA);
    printf("Well done! The output shape is correct.\n");
    printf("gradOutput dims: %d\n", gradOutput.dim());

    printf("gradOutput shape: %d, %d, %d, %d\n", gradOutput.size(0), gradOutput.size(1), gradOutput.size(2), gradOutput.size(3));

    inputCentroids = res[1];
    vector_index = res[2];
    vector_ids = res[3];
    buckets_count = res[4];
    buckets_index = res[5];
    buckets_index_inv = res[6];
    input_row = res[7];

    auto res2 = conv_deep_reuse_backward(
        input_row,
        inputCentroids,
        weights,
        gradOutput,
        vector_index,
        vector_ids,
        buckets_count,
        buckets_index,
        buckets_index_inv,
        random_vectors,
        input_height,
        input_width,
        pad_height,
        pad_width,
        stride_height,
        stride_width,
        param_H,
        alpha,
        sigma,
        do_bias);

    std::cout << "Backward pass finished." << std::endl;
}