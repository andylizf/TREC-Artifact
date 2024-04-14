#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>
#include <driver_types.h>

#include "convDR_backward.h"
#include "convDR_backward_kernel.cuh"

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)
[[__gnu__::__always_inline__]] inline void CHECK_SHAPE(const at::Tensor& tensor, const at::IntArrayRef shape)
{
    TORCH_CHECK(tensor.sizes() == shape, "Expected shape ", shape, " but got ", tensor.sizes());
}

struct BackwardOutput {
    at::Tensor gradInput;
    at::Tensor gradWeights;
    at::Tensor gradHash;
    at::Tensor gradBias; // Optional
};

class CovDeepReuseBackward {
private:
    at::Tensor input_row; // [n_matrices, num_rows, param_L]
    at::Tensor inputCentroids; // [n_matrices, max_buckets, param_L]
    at::Tensor weights; // [nOutputPlane, nInputPlane, kernel_height, kernel_width]
    at::Tensor gradOutput; // [batch_size, nOutputPlane, output_height, output_width]
    at::Tensor vector_index; // [n_matrices, num_rows]
    at::Tensor vector_ids; // [n_matrices, num_rows]
    at::Tensor buckets_count; // [n_matrices, max_buckets]
    at::Tensor buckets_index; // [n_matrices, total_buckets]
    at::Tensor buckets_index_inv; // [n_matrices, max_buckets]
    at::Tensor random_vectors; // [param_L, param_H]
    int64_t input_height;
    int64_t input_width;
    int64_t pad_height;
    int64_t pad_width;
    int64_t stride_height;
    int64_t stride_width;
    int64_t param_H;
    float alpha;
    float sigma;
    bool do_bias;

    int64_t n_matrices;
    int64_t param_L;
    int max_buckets;
    int nOutputPlane;
    int nInputPlane, kernel_height, kernel_width;
    int num_rows;
    int total_buckets;
    int batch_size;
    int output_height, output_width;
    at::IntArrayRef weights_sizes;

    std::pair<at::Tensor, at::Tensor> get_gradParameters(
        cudaStream_t& stream,
        const at::Tensor& gradOutput_centroids // {n_matrices, max_buckets, nOutputPlane}
    ) const
    {
        at::Tensor inputCentroids_col = inputCentroids.transpose(1, 2).contiguous();
        CHECK_SHAPE(inputCentroids_col, { n_matrices, param_L, max_buckets });

        at::Tensor gradWeights = inputCentroids_col.bmm(gradOutput_centroids)
                                     .reshape({ -1, nOutputPlane })
                                     .transpose(0, 1)
                                     .reshape(weights_sizes);
        CHECK_SHAPE(gradWeights, { nOutputPlane, nInputPlane, kernel_height, kernel_width });

        at::Tensor gradBias = do_bias ? gradOutput_centroids[0].sum(0) : at::Tensor();
        return { gradWeights, gradBias };
    }

    std::pair<at::Tensor, at::Tensor> get_gradInput(
        cudaStream_t& stream,
        const at::Tensor& gradOutput_centroids // {n_matrices, max_buckets, nOutputPlane}
    )
    {
        at::Tensor gradInput_rows = at::zeros({ num_rows, n_matrices * param_L }, gradOutput_centroids.options());
        at::Tensor weights_matrices = weights.reshape({ nOutputPlane, n_matrices, param_L }).transpose(0, 1).contiguous();
        at::Tensor gradInput_centroids = gradOutput_centroids.bmm(weights_matrices);
        reconstruct_gradInputRows_cuda(stream, vector_index, gradInput_centroids, gradInput_rows);

        at::Tensor gradInputs = at::zeros({ batch_size, nInputPlane, input_height, input_width },
            gradOutput_centroids.options());
        row2im_batch_cuda(stream, gradInput_rows, gradInputs,
            kernel_height, kernel_width,
            pad_height, pad_width,
            stride_height, stride_width);
        return { std::move(gradInputs), std::move(gradInput_centroids) };
    }

    at::Tensor get_gradHash(
        cudaStream_t& stream,
        const at::Tensor& input_matrix,
        const at::Tensor& hash_bits,
        const at::Tensor& gradIndex)
    {
        at::Tensor grad_Hash_value = (vector_ids.unsqueeze(2).repeat({ 1, 1, max_buckets }) + 1).to(gradIndex.options()) / (buckets_index_inv.unsqueeze(1).repeat({ 1, num_rows, 1 }) + 1).to(gradIndex.options()) - 1;
        grad_Hash_value = -1 * grad_Hash_value / (sigma * sigma) * exp(-1 * grad_Hash_value * grad_Hash_value / (2 * sigma * sigma)) * gradIndex / buckets_count.unsqueeze(1).repeat({ 1, num_rows, 1 }).to(gradIndex.options());
        at::Tensor power = at::zeros({ n_matrices, max_buckets, param_H }, gradIndex.options());
        at::Tensor zero = at::zeros({ n_matrices, num_rows, max_buckets }, gradIndex.options());
        grad_Hash_value = at::where(grad_Hash_value.isnan(), zero, grad_Hash_value);

        get_Power(stream, buckets_index, power, max_buckets, param_H);
        at::Tensor gradHash = grad_Hash_value.bmm(power).reshape({ num_rows * n_matrices, param_H });

        gradHash = (alpha * hash_bits * (1 - hash_bits)) * gradHash;

        return input_matrix.transpose(0, 1).mm(gradHash);
    }

    at::Tensor get_gradOutputSum(cudaStream_t& stream)
    {
        at::Tensor gradOutput_mat = gradOutput.reshape({ batch_size, nOutputPlane,
                                                           output_height * output_width })
                                        .transpose(1, 2)
                                        .reshape({ -1, nOutputPlane });
        CHECK_SHAPE(gradOutput_mat, { batch_size * output_height * output_width, nOutputPlane });
        at::Tensor gradOutput_centroids = at::zeros({ n_matrices, max_buckets, nOutputPlane }, gradOutput.options());
        get_gradOutputCentroids_add_cuda(stream, vector_index, gradOutput_mat, gradOutput_centroids);

        return gradOutput_centroids;
    }

public:
    CovDeepReuseBackward(
        const at::Tensor& input_row,
        const at::Tensor& inputCentroids,
        const at::Tensor& weights,
        const at::Tensor& gradOutput,
        const at::Tensor& vector_index,
        const at::Tensor& vector_ids,
        const at::Tensor& buckets_count,
        const at::Tensor& buckets_index,
        const at::Tensor& buckets_index_inv,
        const at::Tensor& random_vectors,
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
        : input_row(input_row)
        , inputCentroids(inputCentroids)
        , weights(weights)
        , gradOutput(gradOutput)
        , vector_index(vector_index)
        , vector_ids(vector_ids)
        , buckets_count(buckets_count)
        , buckets_index(buckets_index)
        , buckets_index_inv(buckets_index_inv)
        , random_vectors(random_vectors)
        , input_height(input_height)
        , input_width(input_width)
        , pad_height(pad_height)
        , pad_width(pad_width)
        , stride_height(stride_height)
        , stride_width(stride_width)
        , param_H(param_H)
        , alpha(alpha)
        , sigma(sigma)
        , do_bias(do_bias)
        , n_matrices(input_row.size(0))
        , param_L(input_row.size(2))
        , max_buckets(inputCentroids.size(1))
        , nOutputPlane(weights.size(0))
        , nInputPlane(weights.size(1))
        , kernel_height(weights.size(2))
        , kernel_width(weights.size(3))
        , num_rows(vector_index.size(1))
        , total_buckets(buckets_index.size(1))
        , batch_size(gradOutput.size(0))
        , output_height(gradOutput.size(2))
        , output_width(gradOutput.size(3))
        , weights_sizes(weights.sizes())
    {
        CHECK_INPUT(input_row);
        CHECK_SHAPE(input_row, { n_matrices, num_rows, param_L });
        CHECK_INPUT(inputCentroids);
        CHECK_SHAPE(inputCentroids, { n_matrices, max_buckets, param_L });
        CHECK_INPUT(weights);
        CHECK_SHAPE(weights, { nOutputPlane, nInputPlane, kernel_height, kernel_width });
        CHECK_INPUT(gradOutput);
        CHECK_SHAPE(gradOutput, { batch_size, nOutputPlane, output_height, output_width });
        CHECK_INPUT(vector_index);
        CHECK_SHAPE(vector_index, { n_matrices, num_rows });
        CHECK_INPUT(vector_ids);
        CHECK_SHAPE(vector_ids, { n_matrices, num_rows });
        CHECK_INPUT(buckets_count);
        CHECK_SHAPE(buckets_count, { n_matrices, max_buckets });
        CHECK_INPUT(buckets_index);
        CHECK_SHAPE(buckets_index, { n_matrices, total_buckets });
        CHECK_INPUT(buckets_index_inv);
        CHECK_SHAPE(buckets_index_inv, { n_matrices, max_buckets });
        CHECK_INPUT(random_vectors);
        CHECK_SHAPE(random_vectors, { param_L, param_H });
    }

    BackwardOutput backward()
    {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        at::Tensor gradOutput_centroids = get_gradOutputSum(stream);

        const auto& [gradWeights, gradBias] = get_gradParameters(stream, gradOutput_centroids);

        get_gradOutputCentroids_div_cuda(stream, gradOutput_centroids, buckets_count); // ? Not before get_gradParameters?

        const auto& [gradInput, gradInput_centroids] = get_gradInput(stream, gradOutput_centroids);

        at::Tensor gradIndex = input_row.bmm(gradInput_centroids.transpose(1, 2));
        at::Tensor input_matrix = input_row.reshape({ n_matrices * num_rows, param_L });
        at::Tensor hash_bits = 1 / (1 + exp(-1 * alpha * (input_matrix.mm(random_vectors) - 0.1 / pow(2, param_H))));
        const auto& gradHash = get_gradHash(stream, input_matrix, hash_bits, gradIndex);

        return { std::move(gradInput), std::move(gradWeights), std::move(gradHash), std::move(gradBias) };
    }
};

std::vector<at::Tensor> conv_deep_reuse_backward(
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
    auto covDeepReuseBackward = CovDeepReuseBackward {
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
    };
    auto [gradInput, gradWeights, gradHash, gradBias] = covDeepReuseBackward.backward();
    if (do_bias) {
        //! gradBias should be first to match the order in the Python code
        return { gradInput, gradWeights, gradBias, gradHash };
    }
    return { gradInput, gradWeights, gradHash };
}