#pragma once
#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <c10/cuda/CUDAGuard.h>
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

// for each kernel
// refer to Pytorch: pytorch-master/aten/src/ATen/native/cuda/im2col.cuh
// split the whole matrix into L sub-matrices
template <typename scalar_t>
__global__ void im2row_DRbatch_cuda_kernel(
    const int64_t n,
    const scalar_t* __restrict__ data_im,
    const int64_t batch_size,
    const int64_t channels,
    const int64_t height, // input_height
    const int64_t width, // input_width
    const int64_t kernel_height,
    const int64_t kernel_width,
    const int64_t pad_height,
    const int64_t pad_width,
    const int64_t stride_height,
    const int64_t stride_width,
    const int64_t height_row, // output_height = (input_height + 2 * pad_height - kernel_height) / stride_height + 1
    const int64_t width_row, // output_width
    const int64_t param_L,
    const int64_t n_matrices,
    scalar_t* __restrict__ data_row) // = input_row = output
{

    CUDA_1D_KERNEL_LOOP(index, n)
    {
        int64_t w_out = index % width_row;
        int64_t idx = index / width_row;
        int64_t h_out = idx % height_row;
        idx = idx / height_row;
        int64_t channel_in = idx % channels;
        int64_t batch_id = idx / channels;

        int64_t h_in = h_out * stride_height - pad_height;
        int64_t w_in = w_out * stride_width - pad_width;

        const scalar_t* __restrict__ im = data_im + (((batch_id * channels + channel_in) * height + h_in) * width + w_in);

// one thread is responsible for one vector with size kh*kw
#pragma unroll
        for (int i = 0; i < kernel_height; ++i) {
            for (int j = 0; j < kernel_width; ++j) {
                int64_t h = h_in + i;
                int64_t w = w_in + j;
                int64_t row_offset = (channel_in * kernel_height + i) * kernel_width + j;
                int64_t matrix_offset = row_offset % param_L;
                int64_t matrix_id = row_offset / param_L;

                data_row[(((matrix_id * batch_size + batch_id) * height_row + h_out) * width_row + w_out) * param_L + matrix_offset]
                    = (h >= 0 && w >= 0 && h < height && w < width)
                    ? im[i * width + j]
                    : static_cast<scalar_t>(0);
            }
        }
    }
}

// forall i in [0, num_vectors)
//     vector_ids[i] = to_binary(hashed_vectors[i][j] > 0)
//     buckets_count[matrix_id][vector_ids[i]] += 1
template <typename scalar_t>
__global__ void get_id_count_cuda_kernel(
    const int64_t num_vectors,
    const int64_t vector_len,
    const int64_t num_rows,
    const int64_t total_buckets,
    const scalar_t* __restrict__ hashed_vectors,
    int* __restrict__ buckets_count,
    ID_DATATYPE* vector_ids)
{
    const int matrix_id = blockIdx.x;

    // cache buckets_count[matrix_id] in shared memory
    __shared__ extern int shared_buckets_count[]; // total_buckets
    for (int i = threadIdx.x; i < total_buckets; i += blockDim.x) {
        shared_buckets_count[i] = 0;
    }
    __syncthreads();

#pragma unroll 4
    for (int i = threadIdx.x; i < num_rows; i += blockDim.x) {
        const int global_id = matrix_id * num_rows + i;
        const scalar_t* __restrict__ vector_ptr = hashed_vectors + global_id * vector_len;
        ID_DATATYPE id = 0; // a vector_len bit integer
#pragma unroll 4
        for (int j = 0; j < vector_len; j++) {
            id = (id << 1) | (vector_ptr[j] > 0);
        }
        vector_ids[global_id] = id;
        atomicAdd(shared_buckets_count + id, 1);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < total_buckets; i += blockDim.x) {
        buckets_count[matrix_id * total_buckets + i] = shared_buckets_count[i];
    }
}

template <typename scalar_t>
__global__ void get_id_count_cuda_kernel_old(
    const int64_t num_vectors,
    const int64_t vector_len,
    const int64_t num_rows,
    const int64_t total_buckets,
    const scalar_t* __restrict__ hashed_vectors,
    int* __restrict__ buckets_count,
    ID_DATATYPE* vector_ids)
{
    CUDA_1D_KERNEL_LOOP(index, num_vectors)
    {
        const scalar_t* __restrict__ vector_ptr = hashed_vectors + index * vector_len;
        ID_DATATYPE id = 0; // a vector_len bit integer
#pragma unroll
        for (int i = 0; i < vector_len; i++) {
            id = (id << 1) | (vector_ptr[i] > 0);
        }
        vector_ids[index] = id;

        int64_t matrix_id = index / num_rows;
        int64_t offset = matrix_id * total_buckets + id;
        atomicAdd(buckets_count + offset, 1);
    }
}

// for matrix_id, vector_idx, vector_offset in num_rows * n_matrices * vect_dim
//   buckets_sum[matrix_id][vector_ids[matrix_id][vector_idx]][vector_offset] += vectors[matrix_id][vector_idx][vector_offset]
// blockIdx.x: n_matrices
// threadIdx.x: num_rows * vect_dim
template <typename scalar_t>
__global__ void get_centroids_add_cuda_kernel(
    const ID_DATATYPE* vector_ids,
    const scalar_t* __restrict__ vectors,
    scalar_t* __restrict__ buckets_sum, // buckets_centoids
    // int* __restrict__ buckets_count,
    const int64_t total_vect, // num_rows * n_matrices
    const int64_t vect_dim, // H
    const int64_t num_rows,
    const int64_t total_buckets)
{
    const int matrix_id = blockIdx.x;

    // cache vector_ids[matrix_id] in shared memory
    __shared__ extern ID_DATATYPE shared_vector_ids[]; // num_rows

    for (int i = threadIdx.x; i < num_rows; i += blockDim.x) {
        shared_vector_ids[i] = vector_ids[matrix_id * num_rows + i];
    }
    __syncthreads();

    for (int i = threadIdx.x; i < num_rows * vect_dim; i += blockDim.x) {
        const int vector_id = i / vect_dim;
        const int vector_offset = i % vect_dim;

        int int_id = shared_vector_ids[vector_id];
        atomicAdd(&buckets_sum[(matrix_id * total_buckets + int_id) * vect_dim + vector_offset],
            vectors[(matrix_id * num_rows + vector_id) * vect_dim + vector_offset]);
    }
}

// for global_id, vector_offset in total_vect * vect_dim
//   buckets_sum[matrix_id][vector_ids[global_id]][vector_offset] += vectors[global_id][vector_offset]
template <typename scalar_t>
__global__ void get_centroids_add_cuda_kernel_old(
    const ID_DATATYPE* vector_ids,
    const scalar_t* __restrict__ vectors,
    scalar_t* __restrict__ buckets_sum, // buckets_centoids
    // int* __restrict__ buckets_count,
    const int64_t total_vect, // num_rows * n_matrices
    const int64_t vect_dim, // H
    const int64_t num_rows,
    const int64_t total_buckets)
{
    CUDA_1D_KERNEL_LOOP(index, total_vect * vect_dim)
    {
        int64_t vect_offset = index % vect_dim;
        int64_t global_id = index / vect_dim;
        ID_DATATYPE int_id = vector_ids[global_id];
        int64_t matrix_id = global_id / num_rows;
        int64_t offset = matrix_id * total_buckets + int_id;
        int64_t buck_loc = offset * vect_dim + vect_offset;
        atomicAdd(buckets_sum + buck_loc, vectors[index]); // just sum
    }
}

#ifndef PRINT_STATS
// blockIdx.x: n_matrices
// threadIdx.x: total_buckets
__global__ void index_bucket_cuda_kernel(
    const int64_t n_matrices,
    const int64_t total_buckets, // 2^H
    const int* __restrict__ buckets_count, // # of vectors of each bucket
    int* __restrict__ buckets_index,
    int* __restrict__ max_buckets)
{ // sum and max of buckets_num
    int matrix_id = blockIdx.x;

    // reindex buckets_index
    __shared__ int bucket_num;
    if (threadIdx.x == 0) {
        bucket_num = 0;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < total_buckets; i += blockDim.x) {
        int bucket_id = matrix_id * total_buckets + i;
        if (buckets_count[bucket_id] > 0) {
            buckets_index[bucket_id] = atomicAdd(&bucket_num, 1);
        } else {
            buckets_index[bucket_id] = -1;
        }
    }

    if (threadIdx.x % warpSize == 0) {
        atomicMax(max_buckets, bucket_num);
    }
}

// for each submatrix
__global__ void index_bucket_cuda_kernel2(
    const int64_t n_matrices,
    const int64_t total_buckets, // 2^H
    const int* __restrict__ buckets_count, // # of vectors of each bucket
    int* __restrict__ buckets_index,
    // int* __restrict__ buckets_num,             // # of buckets of each submatrix
    int* __restrict__ max_buckets)
{ // sum and max of buckets_num

    CUDA_1D_KERNEL_LOOP(index, n_matrices)
    {
        int bucket_num = 0;
#pragma unroll
        for (int i = 0; i < total_buckets; i++) {
            int64_t bucket_id = index * total_buckets + i;
            if (buckets_count[bucket_id] > 0) {
                buckets_index[bucket_id] = bucket_num++;
            } else {
                buckets_index[bucket_id] = -1;
            }
        }
        atomicMax(max_buckets, bucket_num);
    }
}
#else
// for each submatrix
__global__ void index_bucket_cuda_kernel(
    const int64_t n_matrices,
    const int64_t total_buckets, // 2^H
    const int* __restrict__ buckets_count, // # of vectors of each bucket
    int* __restrict__ buckets_index,
    // int* __restrict__ buckets_num,             // # of buckets of each submatrix
    int* __restrict__ buckets_stats)
{ // sum and max of buckets_num

    CUDA_1D_KERNEL_LOOP(index, n_matrices)
    {
        int bucket_num = 0;
#pragma unroll
        for (int i = 0; i < total_buckets; i++) {
            int64_t bucket_id = index * total_buckets + i;
            if (buckets_count[bucket_id] > 0) {
                buckets_index[bucket_id] = bucket_num++;
            } else {
                buckets_index[bucket_id] = -1;
            }
        }
        // buckets_num[index] = bucket_num;
        atomicAdd(&buckets_stats[0], bucket_num);
        atomicMax(&buckets_stats[1], bucket_num);
    }
}
#endif

// for each vector
__global__ void get_vector_index_cuda_kernel(
    const int64_t n_matrices,
    const int64_t num_rows,
    const int64_t total_buckets,
    const ID_DATATYPE* vector_ids,
    const int* __restrict__ buckets_index,
    int* __restrict__ vector_index)
{

    CUDA_1D_KERNEL_LOOP(global_id, n_matrices * num_rows)
    {
        int64_t matrix_id = global_id / num_rows;
        ID_DATATYPE vect_id = vector_ids[global_id];
        int64_t vect_offset = matrix_id * total_buckets + vect_id;
        int buck_index = buckets_index[vect_offset]; // [n_matrices, total_buckets]
        vector_index[global_id] = buck_index;
    }
}

// for each element, divide centroids
template <typename scalar_t>
__global__ void div_remap_centroids_cuda_kernel(
    const scalar_t* __restrict__ buckets_centroids,
    const int* __restrict__ buckets_index,
    const int* __restrict__ buckets_count,
    scalar_t* __restrict__ centroids_for_compute,
    const int64_t num_buckets, // n_matrices * 2^H
    const int64_t total_buckets, // 2^H
    const int64_t vector_dim,
    const int64_t max_buckets)
{

    CUDA_1D_KERNEL_LOOP(index, num_buckets * vector_dim)
    {
        int64_t global_id = index / vector_dim; // bucket id
        int buck_index = buckets_index[global_id];
        if (buck_index >= 0) {
            int64_t vect_offset = index % vector_dim;
            int64_t matrix_id = global_id / total_buckets;
            int64_t comp_loc = ((matrix_id * max_buckets) + buck_index) * vector_dim + vect_offset;
            int buck_count = buckets_count[global_id];
            centroids_for_compute[comp_loc] = buckets_centroids[index] / buck_count;
        }
    }
}

__global__ void get_buckets_count_out_cuda_kernel(
    const int64_t n_matrices,
    const int64_t total_buckets,
    const int64_t max_buckets,
    const int* __restrict__ buckets_index,
    int* __restrict__ buckets_index_inv,
    const int* __restrict__ buckets_count,
    int* __restrict__ buckets_count_out)
{

    CUDA_1D_KERNEL_LOOP(global_id, n_matrices * total_buckets)
    {
        int buck_index = buckets_index[global_id];
        if (buck_index >= 0) { // valid bucket and non-empty
            int64_t matrix_id = global_id / total_buckets;
            int64_t bucket = global_id % total_buckets;
            int64_t out_loc = matrix_id * max_buckets + buck_index;
            buckets_count_out[out_loc] = buckets_count[global_id];
            buckets_index_inv[out_loc] = bucket;
        }
    }
}

// for each element
template <typename scalar_t>
__global__ void reconstruct_output_cuda_kernel(
    const int64_t n, // index of shape [batch_size, outputHeight, outputWidth, n_output_plane]
    const int64_t n_matrices,
    const int64_t image_size, // outputHeight * outputWidth
    const int64_t batch_size,
    const int64_t n_output_plane,
    const int64_t max_buckets,
    const int* __restrict__ vector_index, // [n_matrices, num_rows] where num_rows = batch_size * image_size
    const scalar_t* __restrict__ centroids_after_mm, // [n_matrices, max_buckets, n_output_plane]
    scalar_t* __restrict__ reconstructed_output // [batch_size, n_output_plane, outputHeight, outputWidth]
)
{
    CUDA_1D_KERNEL_LOOP(index, n)
    {
        int64_t k_out = index % n_output_plane; // plane index
        int64_t idx = index / n_output_plane;
        int64_t image_offset = idx % image_size;
        int64_t batch_id = idx / image_size;

        int64_t out_offset = image_offset + image_size * (k_out + n_output_plane * batch_id);
        scalar_t sum = 0.0;
#pragma unroll
        // reconstructed_output[batch_id][k_out][image_offset]
        // = sum_{i=0}^{n_matrices} centroids_after_mm[i][vector_index[i][batch_id][image_offset]][k_out]
        for (int64_t matrix_id = 0; matrix_id < n_matrices; matrix_id++) {
            int64_t global_id = matrix_id * batch_size * image_size + idx;
            int buck_index = vector_index[global_id]; // vector_index[matrix_id][idx]
            int64_t buck_offset = matrix_id * max_buckets + buck_index;
            int64_t in_offset = buck_offset * n_output_plane + k_out;
            sum += centroids_after_mm[in_offset];
        }
        reconstructed_output[out_offset] = sum;
    }
}

// for each element
template <typename scalar_t>
__global__ void bias_add_cuda_kernel(
    const int64_t n, // outN * outK * outH * outW
    const int64_t n_output_plane,
    const int64_t image_size,
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ bias)
{

    CUDA_1D_KERNEL_LOOP(index, n)
    {
        int64_t k_out = (index / image_size) % n_output_plane;
        output[index] += bias[k_out];
    }
}

void im2row_DRbatch_cuda(
    cudaStream_t& stream,
    const at::Tensor& input, // default input is continuous
    at::Tensor& output,
    const int64_t kernel_height,
    const int64_t kernel_width,
    const int64_t pad_height,
    const int64_t pad_width,
    const int64_t stride_height,
    const int64_t stride_width,
    const int64_t param_L)
{

    int64_t batch_size = input.size(0);
    int64_t n_input_plane = input.size(1);
    int64_t input_height = input.size(2);
    int64_t input_width = input.size(3);

    int64_t output_height = (input_height + 2 * pad_height - kernel_height) / stride_height + 1;
    int64_t output_width = (input_width + 2 * pad_width - kernel_width) / stride_width + 1;
    int64_t row_length = n_input_plane * kernel_width * kernel_height;
    int64_t output_length = output_height * output_width;

    int64_t n_matrices = row_length / param_L;

    int64_t num_kernels = batch_size * n_input_plane * output_length; // num_rows * n_input_plane，每个线程处理kw*kh(卷积核)大小，即一个vector

    // Function AT_DISPATCH_FLOATING_TYPES: 将 ATEN 声明的 Tensor 转换成 global function 可以接受的数据
    // 三个参数，第一个参数是输入数据的源类型，第二个参数是操作的标识符（用于报错显示），第三个参数是一个匿名函数。
    // 在匿名函数运行结束后，AT_DISPATCH_FLOATING_TYPES 会将 Float 数组转化为目标类型（运行中的实际类型）数组。
    // split the input matrix into L sub-matrices
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "im2row_cuda", ([&] {
        im2row_DRbatch_cuda_kernel<scalar_t> // 匿名函数中可以使用 scalar_t 代指目标类型
            <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
                num_kernels,
                input.data_ptr<scalar_t>(),
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
                param_L,
                n_matrices,
                output.data_ptr<scalar_t>());
    }));
    AT_CUDA_CHECK(cudaGetLastError());
}

auto get_id_count_cuda(
    cudaStream_t& stream,
    const at::Tensor& hashed_vectors,
    at::Tensor& vector_ids,
    at::Tensor& buckets_count) -> void
{
    // auto [num_vectors, vector_len] = hashed_vectors.sizes(); // [n_matrices * num_rows, H]
    auto num_vectors { hashed_vectors.size(0) };
    auto vector_len { hashed_vectors.size(1) };
    auto n_matrices { vector_ids.size(0) };

    // auto [n_matrices, num_rows] = buckets_count.sizes();
    auto num_rows { vector_ids.size(1) };

    int64_t total_buckets = buckets_count.size(1); // [n_matrices, total_buckets]

    // double t1 = timestamp1();
    AT_DISPATCH_FLOATING_TYPES(hashed_vectors.scalar_type(), "get_id_count_cuda", ([&] {
        get_id_count_cuda_kernel<scalar_t>
            <<<n_matrices, CUDA_NUM_THREADS, total_buckets * sizeof(ID_DATATYPE), stream>>>(
                num_vectors,
                vector_len,
                num_rows,
                total_buckets,
                hashed_vectors.data_ptr<scalar_t>(),
                buckets_count.data_ptr<int>(),
                vector_ids.data_ptr<ID_DATATYPE>());
    }));
    AT_CUDA_CHECK(cudaGetLastError());
}

void get_centroids_add_cuda(
    cudaStream_t& stream,
    const at::Tensor& vector_ids,
    const at::Tensor& vectors, // input_row -- [n_matrices, num_rows]
    at::Tensor& buckets_sum) // buckets_centroids -- [n_matrices, total_buckets, L]
{
    int64_t num_vectors = vectors.size(0);
    int64_t vector_len = vectors.size(1); // L
    int64_t total_buckets = buckets_sum.size(1);
    int64_t n_matrices = vector_ids.size(0);
    int64_t num_rows = vector_ids.size(1);

    AT_DISPATCH_FLOATING_TYPES(vectors.scalar_type(), "get_centroids_add_cuda", ([&] {
        get_centroids_add_cuda_kernel<scalar_t>
            <<<n_matrices, CUDA_NUM_THREADS, num_rows * sizeof(ID_DATATYPE), stream>>>(
                vector_ids.data_ptr<ID_DATATYPE>(),
                vectors.data_ptr<scalar_t>(),
                buckets_sum.data_ptr<scalar_t>(),
                num_vectors,
                vector_len,
                num_rows,
                total_buckets);
    }));
    AT_CUDA_CHECK(cudaGetLastError());
}

void index_bucket_cuda(
    cudaStream_t& stream,
    const at::Tensor& buckets_count,
    at::Tensor& buckets_index,
    at::Tensor& buckets_stats)
{
    int64_t n_matrices = buckets_index.size(0);
    int64_t total_buckets = buckets_index.size(1);

    index_bucket_cuda_kernel<<<n_matrices, CUDA_NUM_THREADS, 0, stream>>>(
        n_matrices,
        total_buckets,
        buckets_count.data_ptr<int>(),
        buckets_index.data_ptr<int>(),
        buckets_stats.data_ptr<int>());
    AT_CUDA_CHECK(cudaGetLastError());
}

void get_vector_index_cuda(
    cudaStream_t& stream,
    const at::Tensor& vector_ids,
    const at::Tensor& buckets_index,
    at::Tensor& vector_index)
{

    int64_t n_matrices = buckets_index.size(0);
    int64_t total_buckets = buckets_index.size(1);
    int64_t num_rows = vector_ids.size(1);

    get_vector_index_cuda_kernel<<<GET_BLOCKS(n_matrices * num_rows), CUDA_NUM_THREADS, 0, stream>>>(
        n_matrices,
        num_rows,
        total_buckets,
        vector_ids.data_ptr<ID_DATATYPE>(),
        buckets_index.data_ptr<int>(),
        vector_index.data_ptr<int>());
    AT_CUDA_CHECK(cudaGetLastError());
}

void div_remap_centroids_cuda(
    cudaStream_t& stream,
    const at::Tensor& buckets_centroids,
    const at::Tensor& buckets_index,
    const at::Tensor& buckets_count,
    at::Tensor& centroids_for_compute)
{

    int64_t total_buckets = buckets_index.size(1);
    int64_t num_buckets = total_buckets * buckets_index.size(0); // total_buckets * n_matrices
    int64_t vector_len = buckets_centroids.size(2); // 2^H
    int64_t max_buckets = centroids_for_compute.size(1);

    AT_DISPATCH_FLOATING_TYPES(buckets_centroids.scalar_type(), "remap_centroids_cuda", ([&] {
        div_remap_centroids_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(num_buckets * vector_len), CUDA_NUM_THREADS, 0, stream>>>(
                buckets_centroids.data_ptr<scalar_t>(),
                buckets_index.data_ptr<int>(),
                buckets_count.data_ptr<int>(),
                centroids_for_compute.data_ptr<scalar_t>(),
                num_buckets,
                total_buckets,
                vector_len,
                max_buckets);
    }));
    AT_CUDA_CHECK(cudaGetLastError());
}

void get_buckets_count_out_cuda(
    cudaStream_t& stream,
    const at::Tensor& buckets_index,
    at::Tensor& buckets_index_inv,
    const at::Tensor& buckets_count,
    at::Tensor& buckets_count_out)
{
    int64_t n_matrices = buckets_index.size(0);
    int64_t total_buckets = buckets_index.size(1);
    int64_t max_buckets = buckets_count_out.size(1);

    get_buckets_count_out_cuda_kernel<<<GET_BLOCKS(n_matrices * total_buckets), CUDA_NUM_THREADS, 0, stream>>>(
        n_matrices,
        total_buckets,
        max_buckets,
        buckets_index.data_ptr<int>(),
        buckets_index_inv.data_ptr<int>(),
        buckets_count.data_ptr<int>(),
        buckets_count_out.data_ptr<int>());
    AT_CUDA_CHECK(cudaGetLastError());
}

void reconstruct_output_cuda(
    cudaStream_t& stream,
    const at::Tensor& vector_index,
    const at::Tensor& centroids_after_mm,
    at::Tensor& reconstructed_output)
{

    int64_t n_matrices = vector_index.size(0); // [n_matrices, num_rows]
    int64_t max_buckets = centroids_after_mm.size(1); // [n_matrices, max_buckets, n_output_plane]
    int64_t n_output_plane = centroids_after_mm.size(2);
    int64_t batch_size = reconstructed_output.size(0);
    int64_t image_size = reconstructed_output.size(2) * reconstructed_output.size(3); // outputHeight * outputWidth

    int64_t total_threads = batch_size * n_output_plane * image_size;

    AT_DISPATCH_FLOATING_TYPES(centroids_after_mm.scalar_type(), "reconstruct_output_cuda", ([&] {
        reconstruct_output_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(total_threads), CUDA_NUM_THREADS, 0, stream>>>(
                total_threads,
                n_matrices,
                image_size,
                batch_size,
                n_output_plane,
                max_buckets,
                vector_index.data_ptr<int>(),
                centroids_after_mm.data_ptr<scalar_t>(),
                reconstructed_output.data_ptr<scalar_t>());
    }));
    AT_CUDA_CHECK(cudaGetLastError());
}

void bias_add_cuda(
    cudaStream_t& stream,
    at::Tensor& output,
    const at::Tensor& bias)
{

    int64_t n_output_plane = output.size(1);
    int64_t image_size = output.size(2) * output.size(3);
    int64_t total_threads = output.size(0) * n_output_plane * image_size;

    AT_DISPATCH_FLOATING_TYPES(output.scalar_type(), "bias_add_cuda", ([&] {
        bias_add_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(total_threads), CUDA_NUM_THREADS, 0, stream>>>(
                total_threads,
                n_output_plane,
                image_size,
                output.data_ptr<scalar_t>(),
                bias.data_ptr<scalar_t>());
    }));
    AT_CUDA_CHECK(cudaGetLastError());
}
