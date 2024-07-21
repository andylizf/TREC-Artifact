#pragma once
#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdlib>
#include <stdio.h>
#include <sys/time.h>

#define CUDA_NUM_THREADS 256

// 表示线程数大于当前grid开启上限时，一直在block中循环线程计算直到完成任务。
// CUDA_1D_KERNEL_LOOP see https://stackoverflow.com/questions/39470116/tensorflow-what-does-index-denote-in-cuda-1d-kernel-loopindex-nthreads-op-us
// is to ensure that even if the number of elements is more than the number of threads, each thread can process multiple elements
// i.e., each thread will be responsible for the elements of index % threads_num == thread_id
#define CUDA_1D_KERNEL_LOOP(i, n)                              \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
         i += blockDim.x * gridDim.x)
double timestamp1()
{
    timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + 1e-6 * t.tv_usec;
}
// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int N)
{
    return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

#define ID_DATATYPE int
#define ID_DATATYPE_AT at::kInt

// refer to Pytorch: pytorch-master/aten/src/ATen/native/cuda/im2col.cuh
// for each convolution kernel, flatten to a row
// after im2col, the shape is
// [batch_size, output_height, output_width, n_input_plane, kernel_length]
// we then split each row into multiple matrices, each contains `param_L` elements, and the shape becomes
// [matrix_id, batch_size, output_height, output_width, param_L]
template <typename scalar_t>
__global__ void im2row_DRbatch_cuda_kernel(
    const int n,
    at::PackedTensorAccessor32<scalar_t, 4, at::RestrictPtrTraits> data_im,
    const int64_t batch_size,
    const int channels,
    const int input_height,
    const int input_width,
    const int kernel_height,
    const int kernel_width,
    const int pad_height,
    const int pad_width,
    const int stride_height,
    const int stride_width,
    const int output_height,
    const int output_width,
    const int vector_dim,
    at::PackedTensorAccessor32<scalar_t, 5, at::RestrictPtrTraits> data_row) // output
{
    CUDA_1D_KERNEL_LOOP(index, batch_size * channels * output_height * output_width)
    {
        const int w_out = index % output_width;
        const int h_out = (index / output_width) % output_height;
        const int channel_in = (index / output_width / output_height) % channels;
        const int batch_id = index / output_width / output_height / channels;

        const int h_in = h_out * stride_height - pad_height;
        const int w_in = w_out * stride_width - pad_width;

        for (int i = 0; i < kernel_height; ++i) {
            for (int j = 0; j < kernel_width; ++j) {
                const int h = h_in + i;
                const int w = w_in + j;

                const int row_offset = (channel_in * kernel_height + i) * kernel_width + j;
                const int matrix_offset = row_offset % vector_dim;
                const int matrix_id = row_offset / vector_dim;

                data_row[matrix_id][matrix_offset][batch_id][h_out][w_out]
                    = (h >= 0 && w >= 0 && h < input_height && w < input_width)
                    ? data_im[batch_id][channel_in][h][w]
                    : static_cast<scalar_t>(0);
            }
        }
    }
}

template <typename scalar_t>
__device__ inline auto binary_mapping(
    const at::TensorAccessor<scalar_t, 1, at::RestrictPtrTraits, signed int> vector,
    const int vector_dim) -> int
{
    int encoded_int = 0;
    for (int i = 0; i < vector_dim; ++i) {
        encoded_int = (encoded_int << 1) | (vector[i] > 0);
    }
    return encoded_int;
}

// for matrix_id, vector_idx in n_matrices * num_rows
//     vector_ids[matrix_id][vector_idx] = to_binary(hashed_vectors[matrix_id][vector_idx])
//     bucket_counts[matrix_id][vector_ids[matrix_id][vector_idx]] += 1
template <typename scalar_t>
__global__ void get_id_count_cuda_kernel(
    const int vector_dim,
    const int64_t num_rows,
    const int64_t num_buckets,
    const at::PackedTensorAccessor32<scalar_t, 3, at::RestrictPtrTraits> hashed_vectors,
    at::PackedTensorAccessor32<int, 2, at::RestrictPtrTraits> bucket_counts,
    at::PackedTensorAccessor32<int, 2, at::RestrictPtrTraits> bucket_ids)
{
    const int matrix_id = blockIdx.x;

    // cache bucket_counts[matrix_id] in shared memory
    __shared__ extern int shared_bucket_counts[]; // num_buckets
    for (int i = threadIdx.x; i < num_buckets; i += blockDim.x) {
        shared_bucket_counts[i] = 0;
    }
    __syncthreads();

    for (int vector_id = threadIdx.x; vector_id < num_rows; vector_id += blockDim.x) {
        int bucket_id = binary_mapping(hashed_vectors[matrix_id][vector_id], vector_dim);
        bucket_ids[matrix_id][vector_id] = bucket_id;
        atomicAdd(&shared_bucket_counts[bucket_id], 1);
    }
    __syncthreads();

    for (int bucket_id = threadIdx.x; bucket_id < num_buckets; bucket_id += blockDim.x) {
        bucket_counts[matrix_id][bucket_id] = shared_bucket_counts[bucket_id];
    }
}

// for matrix_id, vector_idx, vector_offset in num_rows * n_matrices * vector_dim
//   buckets_sum[matrix_id][vector_ids[matrix_id][vector_idx]][vector_offset]
//      += vectors[matrix_id][vector_idx][vector_offset]
// blockIdx.x: n_matrices
// threadIdx.x: num_rows * vector_dim
template <typename scalar_t>
__global__ void get_centroids_add_cuda_kernel(
    const at::PackedTensorAccessor32<int, 2, at::RestrictPtrTraits> bucket_compact_ids,
    const at::PackedTensorAccessor32<scalar_t, 3, at::RestrictPtrTraits> vectors,
    at::PackedTensorAccessor32<scalar_t, 3, at::RestrictPtrTraits> bucket_sum,
    const int max_bucket,
    const int vector_dim,
    const int num_rows,
    const int batch_size)
{
    const int matrix_id = blockIdx.x;
    const int batch_id = blockIdx.y;
    const int batch_start = batch_id * batch_size;
    const int batch_len = std::min(batch_size, num_rows - batch_start);

    // cache shared_bucket_compact_ids[matrix_id] in shared memory
    __shared__ extern int shared_bucket_compact_ids[]; // batch_len
    for (int i = threadIdx.x; i < batch_len; i += blockDim.x) {
        shared_bucket_compact_ids[i] = bucket_compact_ids[matrix_id][i + batch_start];
    }

    // __shared__ extern float shared_bucket_sum[]; // max_bucket * vector_dim
    // for (int i = threadIdx.x; i < max_bucket * vector_dim; i += blockDim.x) {
    //     shared_bucket_sum[i] = 0;
    // }

    __syncthreads();

    for (int i = threadIdx.x; i < batch_len * vector_dim; i += blockDim.x) {
        const int local_vector_id = i / vector_dim;
        const int vector_offset = i % vector_dim;

        int bucket_compact_id = shared_bucket_compact_ids[local_vector_id];

        atomicAdd(&bucket_sum[matrix_id][bucket_compact_id][vector_offset],
            vectors[matrix_id][batch_start + local_vector_id][vector_offset]);
        // atomicAdd(&shared_bucket_sum[bucket_compact_id * vector_dim + vector_offset],
        //     vectors[matrix_id][batch_start + local_vector_id][vector_offset]);
    }
    __syncthreads();

    // for (int i = threadIdx.x; i < max_bucket * vector_dim; i += blockDim.x) {
    //     atomicAdd(&bucket_sum[matrix_id][i / vector_dim][i % vector_dim], shared_bucket_sum[i]);
    // }
    // __syncthreads();
}

__global__ void index_bucket_cuda_kernel(
    const int64_t num_buckets,
    const at::PackedTensorAccessor32<int, 2, at::RestrictPtrTraits> bucket_counts,
    at::PackedTensorAccessor32<int, 2, at::RestrictPtrTraits> bucket_compact_mapping,
    int* __restrict__ max_buckets)
{
    int matrix_id = blockIdx.x;

    __shared__ int bucket_num;
    if (threadIdx.x == 0) {
        bucket_num = 0;
    }
    __syncthreads();

    for (int bucket_id = threadIdx.x; bucket_id < num_buckets; bucket_id += blockDim.x) {
        if (bucket_counts[matrix_id][bucket_id] > 0) {
            bucket_compact_mapping[matrix_id][bucket_id] = atomicAdd(&bucket_num, 1);
        } else {
            bucket_compact_mapping[matrix_id][bucket_id] = -1;
        }
    }
    __syncthreads();

    if (threadIdx.x % warpSize == 0) {
        atomicMax(max_buckets, bucket_num);
    }
}

__global__ void get_bucket_compact_ids_cuda_kernel(
    const int n_matrices,
    const int num_rows,
    const at::PackedTensorAccessor32<int, 2, at::RestrictPtrTraits> bucket_ids,
    const at::PackedTensorAccessor32<int, 2, at::RestrictPtrTraits> bucket_compact_mapping,
    at::PackedTensorAccessor32<int, 2, at::RestrictPtrTraits> bucket_compact_ids)
{
    CUDA_1D_KERNEL_LOOP(global_id, n_matrices * num_rows)
    {
        int matrix_id = global_id / num_rows;
        int vector_id = global_id % num_rows;
        bucket_compact_ids[matrix_id][vector_id]
            = bucket_compact_mapping[matrix_id][bucket_ids[matrix_id][vector_id]];
    }
}

template <typename scalar_t>
__global__ void div_remap_centroids_cuda_kernel(
    at::PackedTensorAccessor32<scalar_t, 3, at::RestrictPtrTraits> bucket_centroids,
    const at::PackedTensorAccessor32<int, 2, at::RestrictPtrTraits> bucket_compact_mapping,
    const at::PackedTensorAccessor32<int, 2, at::RestrictPtrTraits> bucket_counts,
    const int n_matrices,
    const int num_buckets,
    const int vector_dim)
{
    CUDA_1D_KERNEL_LOOP(index, n_matrices * num_buckets * vector_dim)
    {
        int matrix_id = index / (num_buckets * vector_dim);
        int bucket_id = (index / vector_dim) % num_buckets;
        int vector_offset = index % vector_dim;

        if (int bucket_compact_id = bucket_compact_mapping[matrix_id][bucket_id]; bucket_compact_id >= 0) {
            bucket_centroids[matrix_id][bucket_compact_id][vector_offset] /= bucket_counts[matrix_id][bucket_id];
        }
    }
}

__global__ void get_bucket_counts_out_cuda_kernel(
    const int64_t n_matrices,
    const int64_t num_buckets,
    const at::PackedTensorAccessor32<int, 2, at::RestrictPtrTraits> bucket_compact_mapping,
    at::PackedTensorAccessor32<int, 2, at::RestrictPtrTraits> bucket_compact_mapping_inv,
    const at::PackedTensorAccessor32<int, 2, at::RestrictPtrTraits> bucket_counts,
    at::PackedTensorAccessor32<int, 2, at::RestrictPtrTraits> bucket_counts_out)
{
    CUDA_1D_KERNEL_LOOP(index, n_matrices * num_buckets)
    {
        int matrix_id = index / num_buckets;
        int bucket_id = index % num_buckets;

        if (int bucket_compact_id = bucket_compact_mapping[matrix_id][bucket_id]; bucket_compact_id >= 0) {
            bucket_counts_out[matrix_id][bucket_compact_id] = bucket_counts[matrix_id][bucket_id];
            bucket_compact_mapping_inv[matrix_id][bucket_compact_id] = bucket_id;
        }
    }
}
template <typename scalar_t>
__global__ void reconstruct_output_cuda_kernel(
    const int64_t n,
    const int64_t n_matrices,
    const int64_t image_size,
    const int64_t batch_size,
    const int64_t n_output_plane,
    const at::PackedTensorAccessor32<int, 2, at::RestrictPtrTraits> bucket_compact_ids,
    const at::PackedTensorAccessor32<scalar_t, 3, at::RestrictPtrTraits> centroids_after_mm,
    at::PackedTensorAccessor32<scalar_t, 3, at::RestrictPtrTraits> reconstructed_output)
{
    CUDA_1D_KERNEL_LOOP(index, batch_size * image_size * n_output_plane)
    {
        int batch_id = index / n_output_plane / image_size;
        int image_offset = (index / n_output_plane) % image_size;
        int k_out = index % n_output_plane;

        scalar_t sum = 0.0;
        for (int matrix_id = 0; matrix_id < n_matrices; matrix_id++) {
            int bucket_compact_id = bucket_compact_ids[matrix_id][batch_id * image_size + image_offset];
            sum += centroids_after_mm[matrix_id][bucket_compact_id][k_out];
        }
        reconstructed_output[batch_id][k_out][image_offset] = sum;
    }
}
template <typename scalar_t>
__global__ void bias_add_cuda_kernel(
    const int64_t n,
    const int batch_size,
    const int64_t n_output_plane,
    const int64_t image_size,
    at::PackedTensorAccessor32<scalar_t, 3, at::RestrictPtrTraits> output,
    const at::PackedTensorAccessor32<scalar_t, 1, at::RestrictPtrTraits> bias)
{
    CUDA_1D_KERNEL_LOOP(index, batch_size * n_output_plane * image_size)
    {
        int batch_id = index / n_output_plane / image_size;
        int k_out = (index / image_size) % n_output_plane;
        int image_offset = index % image_size;

        output[batch_id][k_out][image_offset] += bias[k_out];
    }
}
