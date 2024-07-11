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

using at::Tensor;

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

    void im2row_DRbatch_cuda(
        const cudaStream_t stream,
        const Tensor input,
        Tensor output)
    {
        int64_t num_kernels = num_rows * row_length / kernel_length;
        assert(num_kernels == batch_size * n_input_plane * output_height * output_width);

        output = output.view({ n_matrices, batch_size, output_height, output_width, param_L });

        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "im2row_cuda", ([&] {
            im2row_DRbatch_cuda_kernel<scalar_t>
                <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
                    num_kernels,
                    input.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                    batch_size,
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
                    output.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>());
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
                    num_rows,
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
        const Tensor bucket_sum)
    {
        AT_DISPATCH_FLOATING_TYPES(vectors.scalar_type(), "get_centroids_add_cuda", ([&] {
            get_centroids_add_cuda_kernel<scalar_t>
                <<<n_matrices, CUDA_NUM_THREADS, num_rows * sizeof(ID_DATATYPE), stream>>>(
                    bucket_ids.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
                    vectors.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                    bucket_sum.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                    vector_dim,
                    num_rows);
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
        const torch::Tensor centroids_for_compute)
    {
        AT_DISPATCH_FLOATING_TYPES(bucket_centroids.scalar_type(), "remap_centroids_cuda", ([&] {
            div_remap_centroids_cuda_kernel<scalar_t>
                <<<GET_BLOCKS(n_matrices * num_buckets * vector_dim), CUDA_NUM_THREADS, 0, stream>>>(
                    bucket_centroids.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                    bucket_compact_mapping.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
                    bucket_counts.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
                    centroids_for_compute.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
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

        TORCH_CHECK(param_L == random_vectors.size(0), "Inconsistent parameter L and random vectors");
        TORCH_CHECK(param_H == random_vectors.size(1), "Inconsistent parameter H and random vectors");
    }

    auto forward() -> std::vector<Tensor>
    {
        Tensor input_row = at::zeros({ n_matrices, num_rows, param_L }, inputs.options());
        Tensor bucket_ids = at::zeros({ n_matrices, num_rows }, inputs.options().dtype(ID_DATATYPE_AT));
        Tensor bucket_counts = at::zeros({ n_matrices, num_buckets }, inputs.options().dtype(at::kInt));
        Tensor bucket_centroids = at::zeros({ n_matrices, num_buckets, vector_dim }, inputs.options());
        Tensor reconstructed_output = at::zeros({ batch_size, n_output_plane, image_size }, inputs.options());
        Tensor bucket_compact_ids = at::zeros({ n_matrices, num_rows }, inputs.options().dtype(at::kInt));

        auto stream = c10::cuda::getCurrentCUDAStream().stream();

        im2row_DRbatch_cuda(stream, inputs, input_row);

        Tensor hashed_vectors = at::matmul(input_row, random_vectors);

        get_id_count_cuda(stream, hashed_vectors, bucket_ids, bucket_counts);

        get_centroids_add_cuda(stream, bucket_ids, input_row, bucket_centroids);

        Tensor bucket_compact_mapping = at::zeros({ n_matrices, num_buckets }, inputs.options().dtype(at::kInt));

        Tensor bucket_stats = at::zeros(1, inputs.options().dtype(at::kInt));
        index_bucket_cuda(stream, bucket_counts, bucket_compact_mapping, bucket_stats);

        get_bucket_compact_ids_cuda(stream, bucket_ids, bucket_compact_mapping, bucket_compact_ids);

        int64_t max_buckets = bucket_stats.item<int64_t>();
        Tensor centroids_for_compute = at::zeros({ n_matrices, max_buckets, param_L }, inputs.options());

        div_remap_centroids_cuda(stream, bucket_centroids, bucket_compact_mapping, bucket_counts, centroids_for_compute);

        Tensor weights_matrices = weights.reshape({ n_output_plane, row_length })
                                      .t()
                                      .reshape({ n_matrices, param_L, n_output_plane });

        Tensor centroids_after_mm = centroids_for_compute.bmm(weights_matrices);

        reconstruct_output_cuda(stream, bucket_compact_ids, centroids_after_mm, reconstructed_output);

        if (do_bias) {
            bias_add_cuda(stream, reconstructed_output, bias);
        }

        reconstructed_output = reconstructed_output.view({ batch_size, n_output_plane, output_height, output_width });
        if (is_training) {
            Tensor bucket_counts_out = at::zeros({ n_matrices, max_buckets }, inputs.options().dtype(at::kInt));
            Tensor bucket_compact_mapping_inv = at::zeros({ n_matrices, max_buckets }, inputs.options().dtype(at::kInt));
            get_bucket_counts_out_cuda(stream, bucket_compact_mapping, bucket_compact_mapping_inv, bucket_counts, bucket_counts_out);

            return { std::move(reconstructed_output),
                std::move(centroids_for_compute),
                std::move(bucket_compact_ids),
                std::move(bucket_ids),
                std::move(bucket_counts_out),
                std::move(bucket_compact_mapping),
                std::move(bucket_compact_mapping_inv),
                std::move(input_row) };
        }
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
