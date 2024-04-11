#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <c10/cuda/CUDAGuard.h>
#include <stdio.h>
#include <utility>

#include "convDR_backward.h"
#include "convDR_backward_kernel.cuh"
// #include "func_utilis.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

at::Tensor get_gradOutputSum(
    cudaStream_t& stream,
    const at::Tensor& gradOutput,
    const at::Tensor& vector_index,
    const int64_t max_buckets);

void get_gradInput_rows(
    cudaStream_t& stream,
    const at::Tensor& weights,
    const at::Tensor& gradOutput_centroids,
    const at::Tensor& vector_index,
    const int64_t param_L,
    at::Tensor& gradInput_rows);

at::Tensor get_gradHash(
    cudaStream_t& stream,
    const at::Tensor& vector_index,
    const at::Tensor& buckets_count,
    const at::Tensor& buckets_index,
    const at::Tensor& buckets_index_inv,
    const at::Tensor& input_row,
    const at::Tensor& hash_bits,
    const at::Tensor& gradIndex,
    const int64_t max_buckets,
    const int64_t param_L,
    const int64_t param_H,
    const float sigma,
    const float alpha);

std::vector<at::Tensor> get_gradInput(
    cudaStream_t& stream,
    const at::IntArrayRef output_size,
    const at::Tensor& weights, // {nOutputPlane, nInputPlane, kH, kW}
    const at::Tensor& gradOutput_centroids, // {n_matrices, max_buckets, nOutputPlane}
    const at::Tensor& vector_index, // {n_matrices, num_rows}
    const int64_t input_height,
    const int64_t input_width,
    const int64_t pad_height,
    const int64_t pad_width,
    const int64_t stride_height,
    const int64_t stride_width,
    const int64_t param_L)
{
    int64_t n_matrices = gradOutput_centroids.size(0);

    int64_t batch_size = output_size[0];
    int64_t outputHeight = output_size[2];
    int64_t outputWidth = output_size[3];
    int64_t nOutputPlane = weights.size(0);
    int64_t n_input_plane = weights.size(1);
    int64_t kernel_height = weights.size(2);
    int64_t kernel_width = weights.size(3);
    // int64_t input_height = (outputHeight - 1) * stride_height + kernel_height - 2 * pad_height;
    // int64_t input_width = (outputWidth - 1) * stride_width + kernel_width - 2 * pad_width;

    int64_t num_rows = vector_index.size(1);
    int64_t row_length = n_matrices * param_L;
    at::Tensor gradInput_rows = at::zeros({ num_rows, row_length }, gradOutput_centroids.options());

    at::Tensor weights_matrices = weights.reshape({ nOutputPlane, n_matrices, param_L }).transpose(0, 1).contiguous();
    at::Tensor gradInput_centroids = gradOutput_centroids.bmm(weights_matrices);
    // get gradInput_rows
    reconstruct_gradInputRows_cuda(stream, vector_index, gradInput_centroids, gradInput_rows);

    at::Tensor gradInputs = at::zeros({ batch_size, n_input_plane, input_height, input_width },
        gradOutput_centroids.options());
    // DEBUG_INFO
    row2im_batch_cuda(stream, gradInput_rows, gradInputs,
        kernel_height, kernel_width,
        pad_height, pad_width,
        stride_height, stride_width);
    // DEBUG_INFO
    return { gradInputs, gradInput_centroids };
}

std::vector<at::Tensor> get_gradParameters(
    cudaStream_t& stream,
    const at::IntArrayRef kernel_size,
    const at::Tensor& inputCentroids, // {n_matrices, max_buckets, param_L}, centroids_for_compute
    const at::Tensor& gradOutput_centroids, // {n_matrices, max_buckets, nOutputPlane}
    const at::Tensor& vector_index,
    const bool do_bias)
{
    int64_t nOutputPlane = gradOutput_centroids.size(2);

    at::Tensor inputCentroids_col = inputCentroids.transpose(1, 2).contiguous();
    // DEBUG_INFO
    // * inputCentroids_col = {n_matrices, param_L, max_buckets}

    at::Tensor gradWeights = inputCentroids_col.bmm(gradOutput_centroids).reshape({ -1, nOutputPlane }).transpose(0, 1).reshape(kernel_size);
    // * gradWeights = {nOutputPlane, nInputPlane, kernel_height, kernel_width}

    // DEBUG_INFO
    if (do_bias) {
        at::Tensor gradBias = gradOutput_centroids[0].sum(0); // ? [0]
        return { gradWeights, gradBias };
    }
    // DEBUG_INFO

    return { gradWeights };
}

std::vector<at::Tensor> conv_deep_reuse_backward(
    // const at::IntArrayRef input_size,
    const at::Tensor input_row, // {n_matrices, num_row, param_L}
    const at::Tensor inputCentroids, // {n_matrices, max_buckets, param_L}
    const at::Tensor weights, // {nOutputPlane, nInputPlane, kH, kW}
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
    const bool do_bias)
{

    // printf("conv_deep_reuse_backward\n");
    // printf("input_row = %d, %d, %d\n", input_row.size(0), input_row.size(1), input_row.size(2));
    // printf("inputCentroids = %d, %d, %d\n", inputCentroids.size(0), inputCentroids.size(1), inputCentroids.size(2));
    // printf("weights = %d, %d, %d, %d\n", weights.size(0), weights.size(1), weights.size(2), weights.size(3));
    // printf("gradOutput = %d, %d, %d, %d\n", gradOutput.size(0), gradOutput.size(1), gradOutput.size(2), gradOutput.size(3));
    // printf("vector_index = %d, %d\n", vector_index.size(0), vector_index.size(1));
    // std::cout << "vector_index = " << vector_index << std::endl;
    // printf("vector_ids = %d, %d\n", vector_ids.size(0), vector_ids.size(1));
    // std::cout << "vector_ids = " << vector_ids << std::endl;
    // printf("buckets_count = %d, %d\n", buckets_count.size(0), buckets_count.size(1));
    // std::cout << "buckets_count = " << buckets_count << std::endl;
    // printf("buckets_index = %d, %d\n", buckets_index.size(0), buckets_index.size(1));
    // std::cout << "buckets_index = " << buckets_index << std::endl;
    // printf("buckets_index_inv = %d, %d\n", buckets_index_inv.size(0), buckets_index_inv.size(1));
    // std::cout << "buckets_index_inv = " << buckets_index_inv << std::endl;
    // printf("random_vectors = %d, %d\n", random_vectors.size(0), random_vectors.size(1));
    // printf("input_height = %d\n", input_height);
    // printf("input_width = %d\n", input_width);
    // printf("pad_height = %d\n", pad_height);
    // printf("pad_width = %d\n", pad_width);
    // printf("stride_height = %d\n", stride_height);
    // printf("stride_width = %d\n", stride_width);
    // printf("param_H = %d\n", param_H);
    // printf("alpha = %f\n", alpha);
    // printf("sigma = %f\n", sigma);
    // printf("do_bias = %d\n", do_bias);

    CHECK_INPUT(input_row);
    CHECK_INPUT(gradOutput);
    CHECK_INPUT(weights);
    CHECK_INPUT(inputCentroids);
    CHECK_INPUT(vector_index);
    CHECK_INPUT(vector_ids);
    CHECK_INPUT(buckets_count);
    CHECK_INPUT(buckets_index);
    CHECK_INPUT(buckets_index_inv);
    CHECK_INPUT(random_vectors);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    at::IntArrayRef output_size = gradOutput.sizes();
    at::IntArrayRef kernel_size = weights.sizes();
    int64_t max_buckets = inputCentroids.size(1);
    int64_t param_L = inputCentroids.size(2);
    int64_t num_rows = vector_index.size(1);
    int64_t n_matrices = vector_index.size(0);

    at::Tensor gradOutput_centroids = get_gradOutputSum(stream, gradOutput, vector_index, max_buckets);

    std::vector<at::Tensor> gradParas = get_gradParameters(stream, kernel_size,
        inputCentroids, gradOutput_centroids, vector_index, do_bias);

    get_gradOutputCentroids_div_cuda(stream, gradOutput_centroids, buckets_count); // ? Not before get_gradParameters?
    // std::cout << "gradOutput_centroids = " << gradOutput_centroids[0][0] << std::endl;

    std::vector<at::Tensor> gradInput_info = get_gradInput(stream, output_size,
        weights, gradOutput_centroids, vector_index, input_height, input_width,
        pad_height, pad_width, stride_height, stride_width, param_L);

    // DEBUG_INFO
    at::Tensor gradInput = gradInput_info[0];
    at::Tensor gradInput_centroids = gradInput_info[1];

    at::Tensor gradIndex = input_row.bmm(gradInput_centroids.transpose(1, 2));
    at::Tensor input_matrix = input_row.reshape({ n_matrices * num_rows, param_L });
    at::Tensor hash_bits = 1 / (1 + exp(-1 * alpha * (input_matrix.mm(random_vectors) - 0.1 / pow(2, param_H))));
    auto gradHash = get_gradHash(stream, vector_ids, buckets_count, buckets_index, buckets_index_inv, input_matrix, hash_bits, gradIndex, max_buckets, param_L, param_H, sigma, alpha);

    at::Tensor gradWeights = gradParas[0];
    if (do_bias) {
        at::Tensor gradBias = gradParas[1];
        return { gradInput, gradWeights, gradBias, gradHash };
    }
    return { gradInput, gradWeights, gradHash };
}

at::Tensor get_gradHash(
    cudaStream_t& stream,
    const at::Tensor& vector_ids,
    const at::Tensor& buckets_count,
    const at::Tensor& buckets_index,
    const at::Tensor& buckets_index_inv,
    const at::Tensor& input_matrix,
    const at::Tensor& hash_bits,
    const at::Tensor& gradIndex,
    const int64_t max_buckets,
    const int64_t param_L,
    const int64_t param_H,
    const float sigma,
    const float alpha)
{

    int64_t n_matrices = vector_ids.size(0);
    int64_t num_rows = vector_ids.size(1);

    at::Tensor grad_Hash_value = (vector_ids.unsqueeze(2).repeat({ 1, 1, max_buckets }) + 1).to(gradIndex.options()) / (buckets_index_inv.unsqueeze(1).repeat({ 1, num_rows, 1 }) + 1).to(gradIndex.options()) - 1;
    grad_Hash_value = -1 * grad_Hash_value / (sigma * sigma) * exp(-1 * grad_Hash_value * grad_Hash_value / (2 * sigma * sigma)) * gradIndex / buckets_count.unsqueeze(1).repeat({ 1, num_rows, 1 }).to(gradIndex.options());
    // at::Tensor grad_Hash_value = at::zeros({n_matrices, num_rows, max_buckets}, gradIndex.options());
    // gradIndex = gradIndex / buckets_count.unsqueeze(1).repeat({1, num_rows, 1}).to(gradIndex.options());
    at::Tensor power = at::zeros({ n_matrices, max_buckets, param_H }, gradIndex.options());
    at::Tensor zero = at::zeros({ n_matrices, num_rows, max_buckets }, gradIndex.options());
    grad_Hash_value = at::where(grad_Hash_value.isnan(), zero, grad_Hash_value);

    get_Power(stream, buckets_index, power, max_buckets, param_H);
    // grad_Hash_value = -1 * grad_Hash_value / (sigma * sigma) * exp(-1 * grad_Hash_value * grad_Hash_value / (2 * sigma * sigma)) * gradIndex;
    at::Tensor gradHash = grad_Hash_value.bmm(power).reshape({ num_rows * n_matrices, param_H });

    gradHash = (alpha * hash_bits * (1 - hash_bits)) * gradHash;

    return input_matrix.transpose(0, 1).mm(gradHash);
}

at::Tensor get_gradOutputSum(
    cudaStream_t& stream,
    const at::Tensor& gradOutput, // {batch_sizes, nOutputPlane, outH, outW}
    const at::Tensor& vector_index, // {n_matrices, num_rows}
    const int64_t max_buckets)
{
    int64_t batch_size = gradOutput.size(0);
    int64_t nOutputPlane = gradOutput.size(1);
    int64_t outputHeight = gradOutput.size(2);
    int64_t outputWidth = gradOutput.size(3);
    int64_t n_matrices = vector_index.size(0);

    // [batch_size * outputHeight * outputWidth, nOutputPlane]
    at::Tensor gradOutput_mat = gradOutput.reshape({ batch_size, nOutputPlane,
                                                       outputHeight * outputWidth })
                                    .transpose(1, 2)
                                    .reshape({ -1, nOutputPlane });
    // DEBUG_INFO
    at::Tensor gradOutput_centroids = at::zeros({ n_matrices, max_buckets, nOutputPlane }, gradOutput.options());
    // DEBUG_INFO
    get_gradOutputCentroids_add_cuda(stream, vector_index, gradOutput_mat, gradOutput_centroids);

    return gradOutput_centroids;
}

void get_gradInput_rows(
    cudaStream_t& stream,
    const at::Tensor& weights,
    const at::Tensor& gradOutput_centroids,
    const at::Tensor& vector_index,
    const int64_t param_L,
    at::Tensor& gradInput_rows)
{

    int64_t nOutputPlane = weights.size(0);
    int64_t n_matrices = gradOutput_centroids.size(0);

    at::Tensor weights_matrices = weights.reshape({ nOutputPlane, n_matrices,
                                                      param_L })
                                      .transpose(0, 1)
                                      .contiguous();
    // DEBUG_INFO

    at::Tensor gradInput_centroids = gradOutput_centroids.bmm(weights_matrices);
    // DEBUG_INFO
    reconstruct_gradInputRows_cuda(stream, vector_index, gradInput_centroids, gradInput_rows);
}