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
#include <vector_types.h>

#include "../utils.h"
#include "convDR_forward.h"
#include "convDR_forward_kernel.cuh"

#include <fstream>

using at::Tensor;

constexpr std::size_t ceil_div(std::size_t num, std::size_t denom)
{
    return (num + denom - 1) / denom;
}

class CovDeepReuse {
private:
    Tensor inputs; // [batch_size, nInputPlane, inputHeight, inputWidth]
    int64_t batch_size, n_input_plane, input_height, input_width;
    Tensor weights; // [nOutputPlane, nInputPlane, kernel_height, kernel_width]
    int64_t n_output_plane, kernel_height, kernel_width;
    bool do_bias;
    Tensor bias;
    Tensor random_vectors; // [param_L, param_H]
    int64_t pad_height, pad_width, stride_height, stride_width;
    int64_t param_L, param_H;
    bool is_training;

    int64_t kernel_length;
    int64_t row_length;
    int64_t n_matrices;
    int64_t output_height;
    int64_t output_width;
    int image_size;
    int64_t num_rows;
    int64_t num_buckets;
    int64_t& vector_dim = param_L;

    void im2col_cuda(
        const cudaStream_t stream,
        const Tensor input,
        Tensor output)
    {
        int64_t num_kernels = image_size * row_length / kernel_length;
        assert(num_kernels == n_input_plane * output_height * output_width);

        output = output.view({ n_matrices, param_L, output_height, output_width });

        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "im2row_cuda", ([&] {
            im2col_cuda_kernel<scalar_t>
                <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
                    num_kernels,
                    input.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                    n_input_plane,
                    input_height,
                    input_width,
                    kernel_height,
                    kernel_width,
                    pad_height,
                    pad_width,
                    stride_height,
                    stride_width,
                    output_height,
                    output_width,
                    vector_dim,
                    output.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>());
        }));
        AT_CUDA_CHECK(cudaGetLastError());
    }

    void get_id_count_cuda(
        const cudaStream_t stream,
        const Tensor hashed_vectors,
        const Tensor bucket_ids,
        const Tensor bucket_counts)
    {
        AT_DISPATCH_FLOATING_TYPES(hashed_vectors.scalar_type(), "get_id_count_cuda", ([&] {
            get_id_count_cuda_kernel<scalar_t>
                <<<n_matrices, CUDA_NUM_THREADS, num_buckets * sizeof(ID_DATATYPE), stream>>>(
                    param_H,
                    image_size,
                    num_buckets,
                    hashed_vectors.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                    bucket_counts.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
                    bucket_ids.packed_accessor32<int, 2, torch::RestrictPtrTraits>());
        }));
        AT_CUDA_CHECK(cudaGetLastError());
    }

    void get_centroids_add_cuda(
        const cudaStream_t stream,
        const Tensor bucket_ids,
        const Tensor vectors,
        const Tensor bucket_sum,
        const int num_buckets)
    {
        const int sharedMem = num_buckets * vector_dim * sizeof(float) + image_size * sizeof(int);

        AT_DISPATCH_FLOATING_TYPES(vectors.scalar_type(), "get_centroids_add_cuda", ([&] {
            get_centroids_add_cuda_kernel<scalar_t>
                <<<n_matrices, CUDA_NUM_THREADS, sharedMem, stream>>>(
                    bucket_ids.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
                    vectors.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                    bucket_sum.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                    num_buckets,
                    vector_dim,
                    image_size);
        }));
        AT_CUDA_CHECK(cudaGetLastError());
    }

    void index_bucket_cuda(
        const cudaStream_t stream,
        const Tensor bucket_counts,
        const Tensor bucket_compact_mapping,
        const Tensor bucket_stats)
    {
        index_bucket_cuda_kernel<<<n_matrices, CUDA_NUM_THREADS, 0, stream>>>(
            num_buckets,
            bucket_counts.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
            bucket_compact_mapping.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
            bucket_stats.data_ptr<int>());
        AT_CUDA_CHECK(cudaGetLastError());
    }

    void get_bucket_compact_ids_cuda(
        const cudaStream_t stream,
        const Tensor bucket_ids,
        const Tensor bucket_compact_mapping,
        const Tensor bucket_compact_ids)
    {
        get_bucket_compact_ids_cuda_kernel<<<GET_BLOCKS(n_matrices * num_rows), CUDA_NUM_THREADS, 0, stream>>>(
            n_matrices,
            num_rows,
            bucket_ids.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
            bucket_compact_mapping.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
            bucket_compact_ids.packed_accessor32<int, 2, torch::RestrictPtrTraits>());
        AT_CUDA_CHECK(cudaGetLastError());
    }

    void div_remap_centroids_cuda(
        const cudaStream_t stream,
        const torch::Tensor bucket_centroids,
        const torch::Tensor bucket_compact_mapping,
        const torch::Tensor bucket_counts,
        const torch::Tensor compact_bucket_centroids)
    {
        AT_DISPATCH_FLOATING_TYPES(bucket_centroids.scalar_type(), "remap_centroids_cuda", ([&] {
            div_remap_centroids_cuda_kernel<scalar_t>
                <<<GET_BLOCKS(n_matrices * num_buckets * vector_dim), CUDA_NUM_THREADS, 0, stream>>>(
                    bucket_centroids.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                    bucket_compact_mapping.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
                    bucket_counts.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
                    compact_bucket_centroids.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                    n_matrices,
                    num_buckets,
                    vector_dim);
        }));
        AT_CUDA_CHECK(cudaGetLastError());
    }

    void get_bucket_counts_out_cuda(
        const cudaStream_t stream,
        const torch::Tensor bucket_compact_mapping,
        const torch::Tensor bucket_compact_mapping_inv,
        const torch::Tensor bucket_counts,
        const torch::Tensor bucket_counts_out)
    {
        get_bucket_counts_out_cuda_kernel<<<GET_BLOCKS(n_matrices * num_buckets), CUDA_NUM_THREADS, 0, stream>>>(
            n_matrices,
            num_buckets,
            bucket_compact_mapping.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
            bucket_compact_mapping_inv.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
            bucket_counts.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
            bucket_counts_out.packed_accessor32<int, 2, torch::RestrictPtrTraits>());
        AT_CUDA_CHECK(cudaGetLastError());
    }

    void reconstruct_output_cuda(
        const cudaStream_t stream,
        const torch::Tensor bucket_compact_ids,
        const torch::Tensor centroids_after_mm,
        const torch::Tensor reconstructed_output)
    {
        int64_t total_threads = batch_size * n_output_plane * image_size;

        AT_DISPATCH_FLOATING_TYPES(centroids_after_mm.scalar_type(), "reconstruct_output_cuda", ([&] {
            reconstruct_output_cuda_kernel<scalar_t>
                <<<GET_BLOCKS(total_threads), CUDA_NUM_THREADS, 0, stream>>>(
                    total_threads,
                    n_matrices,
                    image_size,
                    batch_size,
                    n_output_plane,
                    bucket_compact_ids.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
                    centroids_after_mm.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                    reconstructed_output.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>());
        }));
        AT_CUDA_CHECK(cudaGetLastError());
    }

    void bias_add_cuda(
        const cudaStream_t stream,
        const torch::Tensor output,
        const torch::Tensor bias)
    {
        int64_t total_threads = batch_size * n_output_plane * image_size;

        AT_DISPATCH_FLOATING_TYPES(output.scalar_type(), "bias_add_cuda", ([&] {
            bias_add_cuda_kernel<scalar_t>
                <<<GET_BLOCKS(total_threads), CUDA_NUM_THREADS, 0, stream>>>(
                    total_threads,
                    batch_size,
                    n_output_plane,
                    image_size,
                    output.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                    bias.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>());
        }));
        AT_CUDA_CHECK(cudaGetLastError());
    }

public:
    CovDeepReuse(const Tensor inputs,
        const Tensor weights,
        const Tensor bias,
        const Tensor random_vectors,
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
        , n_input_plane(inputs.size(1))
        , input_height(inputs.size(2))
        , input_width(inputs.size(3))
        , weights(weights)
        , n_output_plane(weights.size(0))
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
        , kernel_length(kernel_height * kernel_width)
        , row_length(n_input_plane * kernel_length)
        , n_matrices(row_length / param_L)
        , output_height((input_height + 2 * pad_height - kernel_height) / stride_height + 1)
        , output_width((input_width + 2 * pad_width - kernel_width) / stride_width + 1)
        , image_size(output_height * output_width)
        , num_rows(batch_size * image_size)
        , num_buckets(1ll << param_H)
    {
        CHECK_INPUT(inputs);
        CHECK_INPUT(weights);
        CHECK_INPUT(bias);
        CHECK_INPUT(random_vectors);
        // TORCH_CHECK(param_H <= 64, "paramter H must <= 64");
        TORCH_CHECK(param_H <= 32, "Paramter H must <= 32"); // hash value: int32_t
        TORCH_CHECK(n_input_plane == weights.size(1), "Inconsistent number of input channels and weight channels");

        TORCH_CHECK(row_length % param_L == 0, "Parameter L must be the factor of ", row_length);

        TORCH_CHECK(random_vectors.sizes() == torch::IntArrayRef({ param_L, param_H }), "Random vectors must have the shape of [param_L, param_H]");
    }

    auto forward() -> std::vector<Tensor>
    {
        auto stream = c10::cuda::getCurrentCUDAStream().stream();
        auto int_options = inputs.options().dtype(at::kInt);
        auto float_options = inputs.options();

        Tensor bucket_counts = at::zeros({ n_matrices, num_buckets }, int_options);
        Tensor bucket_centroids = at::zeros({ n_matrices, num_buckets, vector_dim }, float_options);
        Tensor bucket_ids = at::zeros({ batch_size, n_matrices, image_size }, int_options);

        for (auto elt = 0; elt < batch_size; elt++) {
            Tensor input = inputs.select(0, elt);

            Tensor input_row = at::zeros({ n_matrices, param_L, image_size }, float_options);
            im2col_cuda(stream, input, input_row);

            Tensor hashed_vectors = input_row.transpose(1, 2).matmul(random_vectors);
            CHECK_SHAPE(hashed_vectors, { n_matrices, image_size, param_H });

            Tensor bucket_ids_elt = bucket_ids.select(0, elt);
            get_id_count_cuda(stream, hashed_vectors, bucket_ids_elt, bucket_counts);
            get_centroids_add_cuda(stream, bucket_ids_elt, input_row, bucket_centroids, num_buckets);
        }

        Tensor bucket_compact_mapping = at::zeros({ n_matrices, num_buckets }, int_options);
        Tensor bucket_stats = at::zeros(1, int_options);
        index_bucket_cuda(stream, bucket_counts, bucket_compact_mapping, bucket_stats);

        Tensor bucket_compact_ids = at::zeros({ n_matrices, num_rows }, int_options);
        bucket_ids = bucket_ids.transpose(0, 1).reshape({ n_matrices, num_rows }).contiguous();
        get_bucket_compact_ids_cuda(stream, bucket_ids, bucket_compact_mapping, bucket_compact_ids);
        int64_t max_buckets = bucket_stats.item<int64_t>();

        Tensor compact_bucket_centroids = at::zeros({ n_matrices, max_buckets, vector_dim }, float_options);
        div_remap_centroids_cuda(stream, bucket_centroids, bucket_compact_mapping, bucket_counts, compact_bucket_centroids);

        Tensor weights_matrices = weights.reshape({ n_output_plane, row_length })
                                      .t()
                                      .reshape({ n_matrices, param_L, n_output_plane });

        Tensor centroids_after_mm = compact_bucket_centroids.bmm(weights_matrices);

        Tensor reconstructed_output = at::zeros({ batch_size, n_output_plane, image_size }, float_options);
        reconstruct_output_cuda(stream, bucket_compact_ids, centroids_after_mm, reconstructed_output);

        if (do_bias) {
            bias_add_cuda(stream, reconstructed_output, bias);
        }

        reconstructed_output = reconstructed_output.view({ batch_size, n_output_plane, output_height, output_width });

        // if (is_training) {
        //     Tensor bucket_counts_out = at::zeros({ n_matrices, max_buckets }, int_options);
        //     Tensor bucket_compact_mapping_inv = at::zeros({ n_matrices, max_buckets }, int_options);
        //     get_bucket_counts_out_cuda(stream, bucket_compact_mapping, bucket_compact_mapping_inv, bucket_counts, bucket_counts_out);

        //     input_row = input_row.transpose(1, 2).contiguous();
        //     return { std::move(reconstructed_output),
        //         std::move(bucket_centroids),
        //         std::move(bucket_compact_ids),
        //         std::move(bucket_ids),
        //         std::move(bucket_counts_out),
        //         std::move(bucket_compact_mapping),
        //         std::move(bucket_compact_mapping_inv),
        //         std::move(input_row) };
        // }

        return { std::move(reconstructed_output) };
    }
};

auto conv_deep_reuse_forward(
    const Tensor inputs,
    const Tensor weights,
    const Tensor bias,
    const Tensor random_vectors,
    const int64_t pad_height,
    const int64_t pad_width,
    const int64_t stride_height,
    const int64_t stride_width,
    const int64_t param_L,
    const int64_t param_H,
    const bool do_bias,
    const bool is_training) -> std::vector<Tensor>
{
    auto cov_deep_reuse = CovDeepReuse {
        std::move(inputs),
        std::move(weights),
        std::move(bias),
        std::move(random_vectors),
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
