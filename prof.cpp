#include <iostream>
#include <torch/extension.h>

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

    at::Tensor input = torch::rand({1, 1, 28, 28}).to(torch::kCUDA);
    at::Tensor weights = torch::rand({1, 1, 3, 3}).to(torch::kCUDA);
    at::Tensor bias = torch::rand({1}).to(torch::kCUDA);
    at::Tensor random_vectors = torch::rand({3, 3}).to(torch::kCUDA);
    
    int64_t pad_height = 0;
    int64_t pad_width = 0;
    int64_t stride_height = 1;
    int64_t stride_width = 1;
    int64_t param_L = 3;
    int64_t param_H = 3;
    bool do_bias = false;
    bool is_training = true;
    bool print_rc = true;
    conv_deep_reuse_forward(
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

    // at::Tensor input_row = torch::rand({1, 3, 3});
    // at::Tensor inputCentroids = torch::rand({1, 3, 3});
    // at::Tensor gradOutput = torch::rand({1, 1, 28, 28});
    // at::Tensor vector_index = torch::rand({1, 3});
    // at::Tensor vector_ids = torch::rand({1, 3});
    // at::Tensor buckets_count = torch::rand({1, 3});
    // at::Tensor buckets_index = torch::rand({1, 3});
    // at::Tensor buckets_index_inv = torch::rand({1, 3});
    // int64_t input_height = 28;
    // int64_t input_width = 28;
    // float alpha = 0.1;
    // float sigma = 0.1;
    // conv_deep_reuse_backward(
    //     input_row,
    //     inputCentroids,
    //     weights,
    //     gradOutput,
    //     vector_index,
    //     vector_ids,
    //     buckets_count,
    //     buckets_index,
    //     buckets_index_inv,
    //     random_vectors,
    //     input_height,
    //     input_width,
    //     pad_height,
    //     pad_width,
    //     stride_height,
    //     stride_width,
    //     param_H,
    //     alpha,
    //     sigma,
    //     do_bias
    // );
}