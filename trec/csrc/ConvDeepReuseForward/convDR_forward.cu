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

    static constexpr int64_t NUM_STREAMS = 4; // 使用4个流进行流水线
    static constexpr int64_t MIN_BATCH_SIZE = 4; // 每个小批次最小大小

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

    void process_batch(
        const cudaStream_t& stream,
        const int64_t start_idx,
        const int64_t end_idx,
        const Tensor& input_row,
        const Tensor& hashed_vectors,
        const Tensor& bucket_ids_batch,
        Tensor& bucket_counts,
        Tensor& bucket_centroids)
    {
        for (auto elt = start_idx; elt < end_idx; elt++) {
            Tensor input = inputs.select(0, elt);
            Tensor input_row_elt = input_row.select(0, elt - start_idx);
            im2col_cuda(stream, input, input_row_elt);

            Tensor hashed_vectors_elt = hashed_vectors.select(0, elt - start_idx);
            hashed_vectors_elt = input_row_elt.transpose(1, 2).matmul(random_vectors);

            Tensor bucket_ids_elt = bucket_ids_batch.select(0, elt - start_idx);
            get_id_count_cuda(stream, hashed_vectors_elt, bucket_ids_elt, bucket_counts);
            get_centroids_add_cuda(stream, bucket_ids_elt, input_row_elt, bucket_centroids, num_buckets);
        }
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
        double TIMER_t, first_t; // 添加计时器变量
        TIMER_START; // 开始计时
        first_t = TIMER_t;

        // 创建CUDA流和事件
        std::vector<cudaStream_t> streams(NUM_STREAMS);
        std::vector<cudaEvent_t> events(NUM_STREAMS);
        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaStreamCreate(&streams[i]);
            cudaEventCreate(&events[i]);
        }
        TIMER_LAP("Stream initialization");

        auto int_options = inputs.options().dtype(at::kInt);
        auto float_options = inputs.options();

        // 为每个流分配临时存储
        std::vector<Tensor> bucket_counts_per_stream;
        std::vector<Tensor> bucket_centroids_per_stream;
        std::vector<Tensor> bucket_ids_per_stream;

        for (int i = 0; i < NUM_STREAMS; i++) {
            bucket_counts_per_stream.push_back(at::zeros({ n_matrices, num_buckets }, int_options));
            bucket_centroids_per_stream.push_back(at::zeros({ n_matrices, num_buckets, vector_dim }, float_options));
            bucket_ids_per_stream.push_back(at::zeros({ MIN_BATCH_SIZE, n_matrices, image_size }, int_options));
        }
        TIMER_LAP("Tensor allocation");

        // 最终结果
        Tensor bucket_counts = at::zeros({ n_matrices, num_buckets }, int_options);
        Tensor bucket_centroids = at::zeros({ n_matrices, num_buckets, vector_dim }, float_options);
        Tensor bucket_ids = at::zeros({ batch_size, n_matrices, image_size }, int_options);

        // 流水线处理
        for (int64_t batch_start = 0; batch_start < batch_size; batch_start += MIN_BATCH_SIZE * NUM_STREAMS) {
            TIMER_LAP("Batch start");

            for (int64_t stream_idx = 0; stream_idx < NUM_STREAMS; stream_idx++) {
                int64_t start_idx = batch_start + stream_idx * MIN_BATCH_SIZE;
                if (start_idx >= batch_size)
                    continue;

                int64_t end_idx = std::min<int64_t>(start_idx + MIN_BATCH_SIZE, batch_size);

                // 为当前批次分配临时存储
                Tensor input_row = at::zeros({ end_idx - start_idx, n_matrices, param_L, image_size }, float_options);
                Tensor hashed_vectors = at::zeros({ end_idx - start_idx, n_matrices, image_size, param_H }, float_options);

                // 异步处理当前批次
                process_batch(
                    streams[stream_idx],
                    start_idx,
                    end_idx,
                    input_row,
                    hashed_vectors,
                    bucket_ids_per_stream[stream_idx],
                    bucket_counts_per_stream[stream_idx],
                    bucket_centroids_per_stream[stream_idx]);

                // 记录事件
                cudaEventRecord(events[stream_idx], streams[stream_idx]);
            }
            TIMER_LAP("Batch processing");

            // 同步所有流并合并结果
            for (int64_t stream_idx = 0; stream_idx < NUM_STREAMS; stream_idx++) {
                cudaEventSynchronize(events[stream_idx]);

                // 合并结果到最终张量
                bucket_counts += bucket_counts_per_stream[stream_idx];
                bucket_centroids += bucket_centroids_per_stream[stream_idx];

                int64_t start_idx = batch_start + stream_idx * MIN_BATCH_SIZE;
                if (start_idx >= batch_size)
                    continue;

                int64_t end_idx = std::min<int64_t>(start_idx + MIN_BATCH_SIZE, batch_size);
                bucket_ids.narrow(0, start_idx, end_idx - start_idx).copy_(bucket_ids_per_stream[stream_idx].narrow(0, 0, end_idx - start_idx));
            }
            TIMER_LAP("Batch sync and merge");
        }

        // 后续处理
        Tensor bucket_compact_mapping = at::zeros({ n_matrices, num_buckets }, int_options);
        Tensor bucket_stats = at::zeros(1, int_options);
        index_bucket_cuda(streams[0], bucket_counts, bucket_compact_mapping, bucket_stats);
        TIMER_LAP("Index bucket");

        Tensor bucket_compact_ids = at::zeros({ n_matrices, num_rows }, int_options);
        bucket_ids = bucket_ids.transpose(0, 1).reshape({ n_matrices, num_rows }).contiguous();
        get_bucket_compact_ids_cuda(streams[0], bucket_ids, bucket_compact_mapping, bucket_compact_ids);

        int64_t max_buckets = bucket_stats.item<int64_t>();
        Tensor compact_bucket_centroids = at::zeros({ n_matrices, max_buckets, vector_dim }, float_options);
        div_remap_centroids_cuda(streams[0], bucket_centroids, bucket_compact_mapping, bucket_counts, compact_bucket_centroids);

        Tensor weights_matrices = weights.reshape({ n_output_plane, row_length })
                                      .t()
                                      .reshape({ n_matrices, param_L, n_output_plane });

        Tensor centroids_after_mm = compact_bucket_centroids.bmm(weights_matrices);

        Tensor reconstructed_output = at::zeros({ batch_size, n_output_plane, image_size }, float_options);
        reconstruct_output_cuda(streams[0], bucket_compact_ids, centroids_after_mm, reconstructed_output);

        if (do_bias) {
            bias_add_cuda(streams[0], reconstructed_output, bias);
        }

        reconstructed_output = reconstructed_output.view({ batch_size, n_output_plane, output_height, output_width });
        TIMER_LAP("Output reshape");

        // 清理流和事件
        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaStreamDestroy(streams[i]);
            cudaEventDestroy(events[i]);
        }
        TIMER_LAP("Cleanup");

        if (is_training) {
            // 训练模式额外处理
            Tensor bucket_counts_out = at::zeros({ n_matrices, max_buckets }, int_options);
            Tensor bucket_compact_mapping_inv = at::zeros({ n_matrices, max_buckets }, int_options);
            get_bucket_counts_out_cuda(streams[0], bucket_compact_mapping, bucket_compact_mapping_inv, bucket_counts, bucket_counts_out);
            TIMER_LAP("Training mode extra processing");

            // 重新组织input_row
            Tensor input_row = at::zeros({ batch_size, n_matrices, param_L, image_size }, float_options);
            for (int64_t batch_start = 0; batch_start < batch_size; batch_start += MIN_BATCH_SIZE) {
                int64_t end_idx = std::min<int64_t>(batch_start + MIN_BATCH_SIZE, batch_size);
                Tensor input_batch = inputs.narrow(0, batch_start, end_idx - batch_start);
                Tensor input_row_batch = input_row.narrow(0, batch_start, end_idx - batch_start);
                im2col_cuda(streams[0], input_batch, input_row_batch);
            }
            TIMER_LAP("Input row reorganization");

            return {
                std::move(reconstructed_output),
                std::move(bucket_centroids),
                std::move(bucket_compact_ids),
                std::move(bucket_ids),
                std::move(bucket_counts_out),
                std::move(bucket_compact_mapping),
                std::move(bucket_compact_mapping_inv),
                std::move(input_row)
            };
        }

        torch::cuda::synchronize();
        cudaDeviceSynchronize();
        printf("Total forward pass: %f\n", timestamp() - first_t);
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
