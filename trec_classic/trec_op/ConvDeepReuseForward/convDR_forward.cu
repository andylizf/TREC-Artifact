
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <stdio.h>
#include <utility>
#include <ATen/TensorUtils.h> 

#include "convDR_forward.h"
#include "convDR_forward_kernel.cuh"
#include "func_utilis.h"
 
#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


void LSH_projection(
    cudaStream_t &stream, 
    const at::Tensor &input_row,
    at::Tensor &random_vectors,     // [param_L, param_H]
    at::Tensor &vector_ids, 
    at::Tensor &buckets_count,
    const int64_t param_L,
    const int64_t param_H);


void clustering_inputs(
    cudaStream_t &stream, 
    const at::Tensor &inputs,
    at::Tensor &random_vectors,
    at::Tensor &vector_ids,
    at::Tensor &buckets_centroids,
    at::Tensor &buckets_count,
    at::Tensor &input_row,
    const int64_t kernel_height,
    const int64_t kernel_width,
    const int64_t pad_height,
    const int64_t pad_width,
    const int64_t stride_height,
    const int64_t stride_width,
    const int64_t param_L,
    const int64_t param_H,
    const int64_t n_matrices,
    const int64_t num_rows);


std::vector<at::Tensor> preprocess_inputs(
    cudaStream_t &stream, 
    const at::Tensor &inputs,
    at::Tensor &random_vectors,
    const int64_t kernel_height,
    const int64_t kernel_width,
    const int64_t pad_height,
    const int64_t pad_width,
    const int64_t stride_height,
    const int64_t stride_width,
    const int64_t param_L,
    const int64_t param_H,
    bool is_training,
    bool print_rc);


std::vector<at::Tensor> conv_deep_reuse_forward(
        const at::Tensor inputs,
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
        const bool print_rc) {

    CHECK_INPUT(inputs);
    CHECK_INPUT(weights);
    CHECK_INPUT(bias);
    CHECK_INPUT(random_vectors);
    // TORCH_CHECK(param_H <= 64, "paramter H must <= 64");
    TORCH_CHECK(param_H <= 32, "Paramter H must <= 32");  // hash value: int32_t

    int64_t batch_size = inputs.size(0);
    int64_t nInputPlane = inputs.size(1);
    int64_t inputHeight = inputs.size(2);
    int64_t inputWidth = inputs.size(3);

    int64_t nOutputPlane = weights.size(0);
    int64_t kernel_height = weights.size(2);
    int64_t kernel_width = weights.size(3);
    TORCH_CHECK(nInputPlane == weights.size(1), "Inconsistent number of input channels and weight channels");

    int64_t row_length = nInputPlane * kernel_width * kernel_height;
    TORCH_CHECK(row_length % param_L == 0, "Parameter L must be the factor of", row_length);
    int64_t n_matrices = row_length / param_L;

    int64_t outputHeight = (inputHeight + 2 * pad_height - kernel_height) / stride_height + 1;
    int64_t outputWidth = (inputWidth + 2 * pad_width - kernel_width) / stride_width + 1;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // information need for inference/training
    // val: {centroids_for_compute, vector_index, buckets_stats}
    // train: {centroids_for_compute, vector_index, buckets_stats, buckets_count_out}
    std::vector<at::Tensor> centroids_info = preprocess_inputs(stream, inputs, random_vectors, kernel_height, kernel_width, pad_height, pad_width, 
                                                            stride_height, stride_width, param_L, param_H, is_training, print_rc);
     
    at::Tensor centroids_for_compute = centroids_info[0];
    at::Tensor vector_index = centroids_info[1];
    at::Tensor buckets_stats = centroids_info[2];
    int64_t max_buckets = buckets_stats.data_ptr<int>()[1];

    at::Tensor weights_matrices = weights.reshape({nOutputPlane, row_length}).t().reshape({n_matrices, param_L, nOutputPlane});

    // [n_matrices, max_buckets, n_output_plane]
    at::Tensor centroids_after_mm = centroids_for_compute.bmm(weights_matrices);  // batch matrix multiplicatiion

    at::Tensor reconstructed_output = at::zeros({batch_size, nOutputPlane, outputHeight, outputWidth}, inputs.options());
    reconstruct_output_cuda(stream, vector_index, centroids_after_mm, reconstructed_output);

    if (do_bias)
        bias_add_cuda(stream, reconstructed_output, bias);

    if (is_training) { 
        at::Tensor buckets_count = centroids_info[3];
        at::Tensor input_row = centroids_info[4];
        at::Tensor buckets_index = centroids_info[5];
        at::Tensor buckets_index_inv = centroids_info[6];
        at::Tensor vector_ids = centroids_info[7];
        return {reconstructed_output, centroids_for_compute, vector_index, vector_ids, buckets_count, buckets_index, buckets_index_inv, input_row};
    }
    else {
        return {reconstructed_output, centroids_info[3]};
    }
    c10::cuda::CUDACachingAllocator::emptyCache();
}


void LSH_projection( 
        cudaStream_t &stream, 
        const at::Tensor &input_row,    // L sub-matrices, [n_matrices * num_rows, L]
        at::Tensor &random_vectors,     // [param_L, param_H]
        at::Tensor &vector_ids,
        at::Tensor &buckets_count,
        const int64_t param_L,
        const int64_t param_H) {
    // at::Tensor random_vectors = at::empty({param_L, param_H}, input_row.options()).uniform_(-1, 1);
    // at::Tensor random_vectors = at::randn({param_L, param_H}, input_row.options());
    at::Tensor hashed_vectors = input_row.mm(random_vectors);  // matmul -- [n_matrices * num_rows, H]
    get_id_count_cuda(stream, hashed_vectors, vector_ids, buckets_count);  // compute hash value and count for each bucket
} 

 
void clustering_inputs(
        cudaStream_t &stream, 
        const at::Tensor &inputs,
        at::Tensor &random_vectors,
        at::Tensor &vector_ids,
        at::Tensor &buckets_centroids,
        at::Tensor &buckets_count,
        at::Tensor &input_row,
        const int64_t kernel_height,
        const int64_t kernel_width,
        const int64_t pad_height,
        const int64_t pad_width,
        const int64_t stride_height,
        const int64_t stride_width,
        const int64_t param_L,
        const int64_t param_H,
        const int64_t n_matrices,
        const int64_t num_rows) {

    // real order: [n_matrices, batch_size, out_height, out_width, param_L]
    // at::Tensor input_row = at::zeros({n_matrices * num_rows, param_L}, inputs.options());
    // split the input matrix into L-submatrices(input_row)
    im2row_DRbatch_cuda(stream, inputs, input_row, kernel_height, kernel_width, 
                        pad_height, pad_width, stride_height, stride_width, param_L);

    LSH_projection(stream, input_row, random_vectors, vector_ids, buckets_count, param_L, param_H);

    get_centroids_add_cuda(stream, vector_ids, input_row, buckets_centroids);
}


std::vector<at::Tensor> preprocess_inputs(
        cudaStream_t &stream, 
        const at::Tensor &inputs,
        at::Tensor &random_vectors,
        const int64_t kernel_height,
        const int64_t kernel_width,
        const int64_t pad_height,
        const int64_t pad_width,
        const int64_t stride_height,
        const int64_t stride_width,
        const int64_t param_L,
        const int64_t param_H,
        bool is_training,
        bool print_rc) { 

    int64_t batch_size = inputs.size(0);
    int64_t nInputPlane = inputs.size(1);
    int64_t inputHeight = inputs.size(2);
    int64_t inputWidth = inputs.size(3);

    int64_t row_length = nInputPlane * kernel_width * kernel_height;
    int64_t n_matrices = row_length / param_L;  // 

    int64_t outputHeight = (inputHeight + 2 * pad_height - kernel_height) / stride_height + 1;
    int64_t outputWidth = (inputWidth + 2 * pad_width - kernel_width) / stride_width + 1;
    int64_t num_rows = batch_size * outputHeight * outputWidth;  //
    int64_t total_buckets = std::pow(2, param_H);  //

    at::Tensor vector_ids = at::zeros({n_matrices, num_rows}, inputs.options().dtype(ID_DATATYPE_AT));
    at::Tensor buckets_centroids = at::zeros({n_matrices, total_buckets, param_L}, inputs.options());
    // # of vectors of each bucket
    at::Tensor buckets_count = at::zeros({n_matrices, total_buckets}, inputs.options().dtype(at::kInt));
    at::Tensor input_row = at::zeros({n_matrices * num_rows, param_L}, inputs.options());
    clustering_inputs(stream, inputs, random_vectors, vector_ids, buckets_centroids, buckets_count, input_row,
                    kernel_height, kernel_width, pad_height, pad_width, 
                    stride_height, stride_width, param_L, param_H, n_matrices, num_rows);
    // std::cout << vector_ids[0][0] << std::endl;
    input_row = input_row.reshape({n_matrices, num_rows, param_L});
    // std::cout << "input_row.size()=" << input_row.size(0) << ", " << input_row.size(1) << ", " << input_row.size(2) << std::endl;

    at::Tensor buckets_index = at::zeros({n_matrices, total_buckets}, inputs.options().dtype(at::kInt));
    at::Tensor buckets_stats = at::zeros({2}, inputs.options().dtype(at::kInt));
    index_bucket_cuda(stream, buckets_count, buckets_index, buckets_stats);
    buckets_stats = buckets_stats.cpu ();  // total buckets number & max buckets number of each matrices
 
    // std::cout << num_rows << std::endl;  // 78400 
    // if (print_rc) {
        buckets_stats = buckets_stats.cpu();
        int64_t sum_buckets = buckets_stats.data_ptr<int>()[0];
        int64_t num_vectors = num_rows * n_matrices;
        double rc = (double)sum_buckets / (double)num_vectors;
        at::Tensor remain_ratio = at::zeros(1, inputs.options());
        remain_ratio[0] = rc;
        // printf("remaining ratio: %f\n", rc);
        // std::cout << remain_ratio << std::endl;
    // }
 
    // buckets_stats = at::zeros({2}, at::kInt);
    // buckets_stats[1] = 1600;

    int64_t max_buckets = buckets_stats.data_ptr<int>()[1];
    // std::cout << "max_buckets = " << max_buckets << std::endl;
    at::Tensor centroids_for_compute = at::zeros({n_matrices, max_buckets, param_L}, inputs.options());
    div_remap_centroids_cuda(stream, buckets_centroids, buckets_index, buckets_count, centroids_for_compute);
 
    // before: vector -> hash value -> bucket
    // after: vector -> bucket
    at::Tensor vector_index = at::zeros({n_matrices, num_rows}, inputs.options().dtype(at::kInt));
    get_vector_index_cuda(stream, vector_ids, buckets_index, vector_index);
 
    if (is_training) {
        // before: 2^H buckets
        // after: max_buckets        
        // std::cout << "total_buckets=" << buckets_count.size(1) << std::endl;
        // std::cout << "max_buckets=" << max_buckets << std::endl;
        at::Tensor buckets_count_out = at::zeros({n_matrices, max_buckets}, inputs.options().dtype(at::kInt));
        at::Tensor buckets_index_inv = at::zeros({n_matrices, max_buckets}, inputs.options().dtype(at::kInt));
        get_buckets_count_out_cuda(stream, buckets_index, buckets_index_inv, buckets_count, buckets_count_out);
        return {centroids_for_compute, vector_index, buckets_stats, buckets_count_out, input_row, buckets_index, buckets_index_inv, vector_ids};
    }
 
    return {centroids_for_compute, vector_index, buckets_stats, remain_ratio};
}