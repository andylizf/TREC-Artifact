#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/TensorBody.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/logging_is_not_google_glog.h>
#include <cstdint>

#include "../utils.h"
#include "convDR_forward.h"
#include "convDR_forward_kernel.cuh"

class CovDeepReuse {
private:
    at::Tensor inputs; // [batch_size, nInputPlane, inputHeight, inputWidth]
    int64_t batch_size, nInputPlane, inputHeight, inputWidth;
    at::Tensor weights; // [nOutputPlane, nInputPlane, kernel_height, kernel_width]
    int64_t nOutputPlane, kernel_height, kernel_width;
    bool do_bias;
    at::Tensor bias;
    at::Tensor random_vectors; // [param_L, param_H]
    int64_t pad_height, pad_width, stride_height, stride_width;
    int64_t param_L, param_H;
    bool is_training;

    int64_t row_length;
    int64_t n_matrices;
    int64_t outputHeight;
    int64_t outputWidth;
    int64_t num_rows;

    auto LSH_projection(
        cudaStream_t& stream,
        const at::Tensor& input_row, // L sub-matrices, [n_matrices * num_rows, L]
        at::Tensor& vector_ids,
        at::Tensor& buckets_count) -> void
    {
        // at::Tensor random_vectors = at::empty({param_L, param_H}, input_row.options()).uniform_(-1, 1);
        // at::Tensor random_vectors = at::randn({param_L, param_H}, input_row.options());
        at::Tensor hashed_vectors = input_row.mm(random_vectors); // matmul -- [n_matrices * num_rows, H]
        get_id_count_cuda(stream, hashed_vectors, vector_ids, buckets_count); // compute hash value and count for each bucket
    }

public:
    CovDeepReuse(const at::Tensor& inputs,
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
        const bool is_training)
        : inputs(inputs)
        , batch_size(inputs.size(0))
        , nInputPlane(inputs.size(1))
        , inputHeight(inputs.size(2))
        , inputWidth(inputs.size(3))
        , weights(weights)
        , nOutputPlane(weights.size(0))
        , kernel_height(weights.size(2))
        , kernel_width(weights.size(3))
        , do_bias(do_bias)
        , bias(bias)
        , random_vectors(random_vectors)
        , pad_height(pad_height)
        , pad_width(pad_width)
        , stride_height(stride_height)
        , stride_width(stride_width)
        , param_L(param_L)
        , param_H(param_H)
        , is_training(is_training)
        , row_length(nInputPlane * kernel_width * kernel_height)
        , n_matrices(row_length / param_L)
        , outputHeight((inputHeight + 2 * pad_height - kernel_height) / stride_height + 1)
        , outputWidth((inputWidth + 2 * pad_width - kernel_width) / stride_width + 1)
        , num_rows(batch_size * outputHeight * outputWidth)
    {
        CHECK_INPUT(inputs);
        CHECK_INPUT(weights);
        CHECK_INPUT(bias);
        CHECK_INPUT(random_vectors);
        // TORCH_CHECK(param_H <= 64, "paramter H must <= 64");
        TORCH_CHECK(param_H <= 32, "Paramter H must <= 32"); // hash value: int32_t
        TORCH_CHECK(nInputPlane == weights.size(1), "Inconsistent number of input channels and weight channels");

        TORCH_CHECK(row_length % param_L == 0, "Parameter L must be the factor of", row_length);

        TORCH_CHECK(param_L == random_vectors.size(0), "Inconsistent parameter L and random vectors");
        TORCH_CHECK(param_H == random_vectors.size(1), "Inconsistent parameter H and random vectors");
    }

    auto forward() -> std::vector<at::Tensor>
    {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        // * inputs: [batch_size, nInputPlane, inputHeight, inputWidth]

        //! preprocess
        double TIMER_START;
        double t0 = timestamp();

        at::Tensor input_row = at::zeros({ n_matrices * num_rows, param_L }, inputs.options());
        im2row_DRbatch_cuda(stream, inputs, input_row, kernel_height, kernel_width,
            pad_height, pad_width, stride_height, stride_width, param_L);
        // * input_row: [n_matrices * num_rows, L]
        // * the input matrix after im2row

        TIMER_LAP("im2row_DRbatch_cuda");

        at::Tensor vector_ids = at::zeros({ n_matrices, num_rows }, inputs.options().dtype(ID_DATATYPE_AT));
        int64_t total_buckets = std::pow(2, param_H); //
        at::Tensor buckets_count = at::zeros({ n_matrices, total_buckets }, inputs.options().dtype(at::kInt));
        at::Tensor buckets_centroids = at::zeros({ n_matrices, total_buckets, param_L }, inputs.options());

        TIMER_LAP("init vector_ids, buckets_count, buckets_centroids");

        at::Tensor hashed_vectors = input_row.mm(random_vectors); // matmul -- [n_matrices * num_rows, H]

        TIMER_LAP("hashed_vectors");

        get_id_count_cuda(stream, hashed_vectors, vector_ids, buckets_count); // compute hash value and count for each bucket

        TIMER_LAP("get_id_count_cuda");

        // TODO: get max_buckets here

        get_centroids_add_cuda(stream, vector_ids, input_row, buckets_centroids);
        // * vector_ids: [n_matrices, num_rows]
        // * the bucket index of each vector (empty buckets including)
        // * buckets_count: [n_matrices, total_buckets]
        // * the count of each bucket (empty buckets including)

        // * the sum per element of vector in the same bucket

        TIMER_LAP("get_centroids_add_cuda");

        at::Tensor buckets_index = at::zeros({ n_matrices, total_buckets }, inputs.options().dtype(at::kInt));
        at::Tensor buckets_stats = at::zeros({ 2 }, inputs.options().dtype(at::kInt));

        TIMER_LAP("init buckets_index, buckets_stats");

        index_bucket_cuda(stream, buckets_count, buckets_index, buckets_stats);
        // * buckets_index: the uniform index of each bucket (without empty buckets)
        // ! but only the index in its own matrix
        // * total_buckets: the total number of buckets
        // * max_buckets: the max number of buckets in each matrices

        TIMER_LAP("index_bucket_cuda");

        at::Tensor vector_index = at::zeros({ n_matrices, num_rows }, inputs.options().dtype(at::kInt));
        get_vector_index_cuda(stream, vector_ids, buckets_index, vector_index);

        // * vector_index: [n_matrices, num_rows]
        // * the uniform bucket index of each vector (without empty buckets)
        TIMER_LAP("get_vector_index_cuda");

        buckets_stats = buckets_stats.cpu();
        auto buckets_stats_ptr = buckets_stats.data_ptr<int>();
        int64_t max_buckets = buckets_stats_ptr[1];

        // TODO: sync max_buckets here

        TIMER_LAP("max_buckets");

        at::Tensor centroids_for_compute = at::zeros({ n_matrices, max_buckets, param_L }, inputs.options());
        div_remap_centroids_cuda(stream, buckets_centroids, buckets_index, buckets_count, centroids_for_compute);
        // * centroids_for_compute: [n_matrices, max_buckets, L]
        // * the average per element of vector in the same bucket

        TIMER_LAP("div_remap_centroids_cuda");

        int64_t num_vectors = num_rows * n_matrices;
        int64_t sum_buckets = buckets_stats_ptr[0];
        auto remain_ratio = (double)sum_buckets / (double)num_vectors;
        auto remain_ratio_tensor = at::tensor({ remain_ratio }, inputs.options());
        //! end preprocess

        TIMER_LAP("remain_ratio");

        // original: [nOutputPlane, nInputPlane, kernel_height, kernel_width]
        // row_length(nInputPlane * kernel_width * kernel_height)
        // n_matrices(row_length / param_L)
        at::Tensor weights_matrices = weights.reshape({ nOutputPlane, row_length }).t().reshape({ n_matrices, param_L, nOutputPlane });

        // [n_matrices, max_buckets, nOutputPlane]
        at::Tensor centroids_after_mm = centroids_for_compute.bmm(weights_matrices); // batch matrix multiplicatiion
        // * centroids_after_mm: [n_matrices, max_buckets, nOutputPlane]

        TIMER_LAP("centroids_after_mm");

        at::Tensor reconstructed_output = at::zeros({ batch_size, nOutputPlane, outputHeight, outputWidth }, inputs.options());
        reconstruct_output_cuda(stream, vector_index, centroids_after_mm, reconstructed_output);
        // * reconstructed_output: [batch_size, nOutputPlane, outputHeight, outputWidth]
        // * since [batch_size, ]

        TIMER_LAP("reconstruct_output_cuda");

        if (do_bias) {
            bias_add_cuda(stream, reconstructed_output, bias);
            TIMER_LAP("bias_add_cuda");
        }

        if (is_training) {
            at::Tensor buckets_count_out = at::zeros({ n_matrices, max_buckets }, inputs.options().dtype(at::kInt));
            at::Tensor buckets_index_inv = at::zeros({ n_matrices, max_buckets }, inputs.options().dtype(at::kInt));
            get_buckets_count_out_cuda(stream, buckets_index, buckets_index_inv, buckets_count, buckets_count_out);
            // * buckets_count_out: [n_matrices, max_buckets]
            // * the count of each bucket (empty buckets not including)
            // * buckets_index_inv: [n_matrices, max_buckets]
            // * the original index of each bucket (empty buckets not including)

            TIMER_LAP("get_buckets_count_out_cuda");
            DEBUG_PRINT("forward time: %f\n", timestamp() - t0);

            return { std::move(reconstructed_output),
                std::move(centroids_for_compute),
                std::move(vector_index),
                std::move(vector_ids),
                std::move(buckets_count_out),
                std::move(buckets_index),
                std::move(buckets_index_inv),
                input_row.reshape({ n_matrices, num_rows, param_L }) };
        }
        DEBUG_PRINT("forward time: %f\n", timestamp() - t0);
        return { std::move(reconstructed_output),
            std::move(remain_ratio_tensor) };
        // c10::cuda::CUDACachingAllocator::emptyCache();
        // ? Is it necessary to empty the cache?
    }
};

auto conv_deep_reuse_forward(
    const at::Tensor& inputs,
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
    const bool is_training) -> std::vector<at::Tensor>
{
    auto cov_deep_reuse = CovDeepReuse {
        inputs,
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
    };
    return cov_deep_reuse.forward();
}