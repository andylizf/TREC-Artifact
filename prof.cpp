#include <iostream>
#include <torch/extension.h>
#include <cuda.h>

std::vector<at::Tensor> conv_deep_reuse_forward(const at::Tensor input,
                                                const at::Tensor weights,
                                                const at::Tensor bias,
                                                at::Tensor random_vectors,
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
        const at::Tensor input_row,     // {n_matrices, num_row, param_L}
        const at::Tensor inputCentroids,  // {n_matrices, max_buckets, param_L}
        const at::Tensor weights,
        const at::Tensor gradOutput,      // {N, K, outH, outW}
        const at::Tensor vector_index,    // {n_matrices, num_rows}
        const at::Tensor vector_ids,    // {n_matrices, num_rows}
        const at::Tensor buckets_count,   // {n_matrices, max_buckets}
        const at::Tensor buckets_index,   // {n_matrices, total_buckets}
        const at::Tensor buckets_index_inv,   // {n_matrices, max_buckets}
        const at::Tensor random_vectors,  // {L, H}
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

int main()
{
    if (!torch::cuda::is_available()) {
        std::cerr << "CUDA is not available. Please check your configuration." << std::endl;
        return 1;
    }

    std::cout << "This file is used to profile the performance of the ConvDeepReuseBackward and ConvDeepReuseForward." << std::endl;    

// this will fail
    // at::Tensor input = torch::rand({20, 3, 1024, 1024}).to(torch::kCUDA);
    // at::Tensor weights = torch::rand({20, 3, 5, 5}).to(torch::kCUDA);
    // at::Tensor bias = torch::rand({1}).to(torch::kCUDA);
    // at::Tensor random_vectors = torch::rand({5, 5}).to(torch::kCUDA);

    /*
        cifar
        conv_deep_reuse_forward
            inputs.size() = 10, 16, 16, 16
            weights.size() = 64, 16, 3, 3
            bias.size() = 64
            random_vectors.size() = 48, 5
            pad_height = 1, pad_width = 1, stride_height = 1, stride_width = 1
            param_L = 48, param_H = 5
            do_bias = 1, is_training = 1, print_rc = 0
        conv_deep_reuse_backward
            input_row = 3, 2560, 48
            inputCentroids = 3, 32, 48
            weights = 64, 16, 3, 3
            gradOutput = 10, 64, 16, 16
            vector_index = 3, 2560
            vector_ids = 3, 2560
            buckets_count = 3, 32
            buckets_index = 3, 32
            buckets_index_inv = 3, 32
            random_vectors = 48, 5
            input_height = 16
            input_width = 16
            pad_height = 1
            pad_width = 1
            stride_height = 1
            stride_width = 1
            param_H = 5
            alpha = 10000.000000
            sigma = 0.006250
            do_bias = 1  
    */
    constexpr int batch_size = 10240;

    at::Tensor input = torch::rand({batch_size, 16, 16, 16}).to(torch::kCUDA);
    at::Tensor weights = torch::rand({64, 16, 3, 3}).to(torch::kCUDA);
    at::Tensor bias = torch::rand({64}).to(torch::kCUDA);
    at::Tensor random_vectors = torch::rand({48, 5}).to(torch::kCUDA);

    int64_t pad_height = 1;
    int64_t pad_width = 1;
    int64_t stride_height = 1;
    int64_t stride_width = 1;
    int64_t param_L = 48;
    int64_t param_H = 5;
    bool do_bias = true;
    bool is_training = true;
    bool print_rc = false;
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
        print_rc
    );

    at::Tensor input_row = torch::rand({3, 2560, 48}).to(torch::kCUDA);
    at::Tensor inputCentroids = torch::rand({3, 32, 48}).to(torch::kCUDA);
    at::Tensor gradOutput = torch::rand({batch_size, 64, 16, 16}).to(torch::kCUDA);
/*
*/
    auto options = torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA);
    at::Tensor vector_index = torch::tensor(
        #include "/home/lizhifei/TREC-Artifact/vector_index.txt"
    , options);
    at::Tensor vector_ids = torch::tensor(
        #include "/home/lizhifei/TREC-Artifact/vector_ids.txt"
    , options);
    at::Tensor buckets_count = torch::tensor({
        {86, 58, 257, 429, 1, 26, 7, 318, 307, 206, 261, 5, 1, 43, 27, 52, 23, 184, 91, 3, 21, 14, 69, 38, 1, 23, 9, 0, 0, 0, 0, 0},
        {81, 99, 316, 486, 3, 5, 31, 12, 163, 79, 493, 219, 6, 6, 39, 25, 82, 27, 77, 71, 1, 1, 5, 2, 49, 28, 88, 41, 11, 2, 9, 3},
        {70, 119, 177, 190, 8, 28, 30, 37, 117, 273, 417, 174, 33, 70, 69, 19, 123, 55, 249, 140, 8, 3, 1, 4, 33, 46, 31, 24, 2, 2, 6, 2}
    }, options);
    at::Tensor buckets_index = torch::tensor({
        {0, 1, 2, 3, 4, -1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, -1, -1, 19, -1, 20, 21, 22, 23, 24, -1, 25, 26},
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31},
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31}
    }, options);
    at::Tensor buckets_index_inv = torch::tensor({
        {0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 24, 25, 26, 27, 28, 30, 31, 0, 0, 0, 0, 0},
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31},
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31}
    }, options);
    int64_t input_height = 16;
    int64_t input_width = 16;
    float alpha = 10000;
    float sigma = 0.00625;
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
        do_bias
    );
}