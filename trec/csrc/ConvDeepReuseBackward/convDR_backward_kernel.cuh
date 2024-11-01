#pragma once
#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <c10/cuda/CUDAGuard.h>

#define MY_CUDA_CHECK(EXPR)                                                                                                                    \
    do {                                                                                                                                       \
        cudaError_t __err = EXPR;                                                                                                              \
        if (__err != cudaSuccess) {                                                                                                            \
            auto error_unused C10_UNUSED = cudaGetLastError();                                                                                 \
            TORCH_CHECK(false, "CUDA error: ", cudaGetErrorString(__err), "File: ", __FILE__, "Line: ", __LINE__, "Function: ", __FUNCTION__); \
        }                                                                                                                                      \
    } while (0)

#define DEBUG_INFO           \
    cudaDeviceSynchronize(); \
    printf("File: %s, Line: %d, Function: %s\n", __FILE__, __LINE__, __FUNCTION__);

#define CUDA_NUM_THREADS 256

#define CUDA_1D_KERNEL_LOOP(i, n)                              \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
         i += blockDim.x * gridDim.x)

// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int N)
{
    return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

// typedef at::kInt ID_DATATYPE;
#define ID_DATATYPE int
#define ID_DATATYPE_AT at::kInt

template <typename scalar_t>
__global__ void get_gradOutputCentroids_add_cuda_kernel(
    const int* __restrict__ vector_index,
    const scalar_t* __restrict__ gradOutput_mat, // {batch_size * outputHeight * outputWidth, nOutputPlane}
    scalar_t* __restrict__ gradOutput_centroids, // {n_matrices, max_buckets, nOutputPlane}
    const int64_t max_buckets,
    const int64_t n_output_plane,
    const int64_t num_rows, // batch_size * outputHeight * outputWidth
    const int64_t n_matrices)
{
    CUDA_1D_KERNEL_LOOP(global_id, num_rows * n_matrices)
    {
        int vect_index = vector_index[global_id];
        int64_t matrix_id = global_id / num_rows;
        int64_t vec_id = global_id % num_rows;
        int64_t buck_offset = matrix_id * max_buckets + vect_index;

        scalar_t* __restrict__ buck_start = gradOutput_centroids + buck_offset * n_output_plane;
        const scalar_t* __restrict__ vect_start = gradOutput_mat + vec_id * n_output_plane;
        for (int i = 0; i < n_output_plane; i++) {
            atomicAdd(buck_start + i, vect_start[i]);
        }
        // for matrix_id, vec_id in n_matrices * num_rows:
        //   for all i in n_output_plane:
        //     gradOutput_centroids[matrix_id][vect_index[matrix_id][vec_id]][i] += gradOutput_mat[vec_id][i]
    }
}

void get_gradOutputCentroids_add_cuda(
    cudaStream_t& stream,
    const at::Tensor& vector_index,
    const at::Tensor& gradOutput_mat,
    at::Tensor& gradOutput_centroids)
{
    int64_t num_rows = gradOutput_mat.size(0);
    int64_t n_output_plane = gradOutput_mat.size(1);
    int64_t n_matrices = gradOutput_centroids.size(0);
    int64_t max_buckets = gradOutput_centroids.size(1);

    AT_DISPATCH_FLOATING_TYPES(gradOutput_mat.scalar_type(), "get_gradOutputCentroids_add_cuda", ([&] {
        get_gradOutputCentroids_add_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(num_rows * n_matrices), CUDA_NUM_THREADS, 0, stream>>>(
                vector_index.data_ptr<int>(),
                gradOutput_mat.data_ptr<scalar_t>(),
                gradOutput_centroids.data_ptr<scalar_t>(),
                max_buckets,
                n_output_plane,
                num_rows,
                n_matrices);
    }));
    MY_CUDA_CHECK(cudaGetLastError());
}

// for bucket_id, n_output_plane in num_buckets * n_output_plane:
//   gradOutput_centroids[bucket_id][n_output_plane] /= buckets_count[bucket_id]
template <typename scalar_t>
__global__ void get_gradOutputCentroids_div_cuda_kernel(
    scalar_t* __restrict__ gradOutput_centroids,
    const int* __restrict__ buckets_count,
    const int64_t num_buckets, // n_matrices * max_buckets
    const int64_t n_output_plane)
{

    CUDA_1D_KERNEL_LOOP(index, num_buckets * n_output_plane)
    {
        int global_id = index / n_output_plane;
        int buck_count = buckets_count[global_id];
        if (buck_count > 0) {
            gradOutput_centroids[index] /= buck_count;
        }
    }
}

void get_gradOutputCentroids_div_cuda(
    cudaStream_t& stream,
    at::Tensor& gradOutput_centroids, // {n_matrices, max_buckets, nOutputPlane}
    const at::Tensor& buckets_count)
{
    int64_t num_buckets = gradOutput_centroids.size(0) * gradOutput_centroids.size(1);
    int64_t vector_len = gradOutput_centroids.size(2);

    AT_DISPATCH_FLOATING_TYPES(gradOutput_centroids.scalar_type(), "get_gradOutputCentroids_div_cuda", ([&] {
        get_gradOutputCentroids_div_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(num_buckets * vector_len), CUDA_NUM_THREADS, 0, stream>>>(
                gradOutput_centroids.data_ptr<scalar_t>(),
                buckets_count.data_ptr<int>(),
                num_buckets,
                vector_len);
    }));
    MY_CUDA_CHECK(cudaGetLastError());
}

// for matrix_id, vec_id in n_matrices * num_rows:
//   for i in n_output_plane:
//     vect_index = vector_index[matrix_id][vec_id]
//     gradInput_rows[vec_id][matrix_id][i] = gradOutput_centroids[matrix_id][vect_index][i]
// TODO: why not let L be part of thread index?
template <typename scalar_t>
__global__ void reconstruct_gradInputRows_cuda_kernel(
    const int* __restrict__ vector_index,
    const scalar_t* __restrict__ gradInput_centroids,
    scalar_t* __restrict__ gradInput_rows,
    const int64_t max_buckets,
    const int64_t param_L,
    const int64_t num_rows,
    const int64_t n_matrices)
{

    CUDA_1D_KERNEL_LOOP(global_id, num_rows * n_matrices)
    {
        int vect_index = vector_index[global_id];
        int64_t matrix_offset = global_id % num_rows;
        int64_t matrix_id = global_id / num_rows;

        scalar_t* __restrict__ row_start = gradInput_rows + (matrix_offset * n_matrices + matrix_id) * param_L;
        const scalar_t* __restrict__ buck_start = gradInput_centroids + (matrix_id * max_buckets + vect_index) * param_L;

        for (int i = 0; i < param_L; i++) {
            row_start[i] = buck_start[i];
        }
    }
}

void reconstruct_gradInputRows_cuda(
    cudaStream_t& stream,
    const at::Tensor& vector_index,
    const at::Tensor& gradInput_centroids,
    at::Tensor& gradInput_rows)
{

    int64_t num_rows = gradInput_rows.size(0);
    int64_t n_matrices = gradInput_centroids.size(0);
    int64_t max_buckets = gradInput_centroids.size(1);
    int64_t param_L = gradInput_centroids.size(2);

    AT_DISPATCH_FLOATING_TYPES(gradInput_centroids.scalar_type(), "reconstruct_gradInputRows_cuda", ([&] {
        reconstruct_gradInputRows_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(num_rows * n_matrices), CUDA_NUM_THREADS, 0, stream>>>(
                vector_index.data_ptr<int>(),
                gradInput_centroids.data_ptr<scalar_t>(),
                gradInput_rows.data_ptr<scalar_t>(),
                max_buckets,
                param_L,
                num_rows,
                n_matrices);
    }));
    MY_CUDA_CHECK(cudaGetLastError());
}

template <typename scalar_t>
__global__ void row2im_batch_cuda_kernel(
    const int64_t n,
    const scalar_t* __restrict__ data_row,
    // const int64_t batch_size,
    const int64_t channels, // nInputPlane
    const int64_t height, // input_height
    const int64_t width, // input_width
    const int64_t kernel_height,
    const int64_t kernel_width,
    const int64_t pad_height,
    const int64_t pad_width,
    const int64_t stride_height,
    const int64_t stride_width,
    const int64_t height_row, // output_height
    const int64_t width_row, // output_width
    scalar_t* __restrict__ data_im)
{

    CUDA_1D_KERNEL_LOOP(index, n)
    {
        int64_t w_im = index % width + pad_width;
        int64_t idx = index / width;
        int64_t h_im = idx % height + pad_height;
        idx = idx / height;
        int64_t c_im = idx % channels;
        int64_t batch_id = idx / channels;

        if (c_im >= channels || h_im >= height || w_im >= width) {
            return;
        }

        int64_t w_row_start = (w_im < kernel_width)
            ? 0
            : (w_im - kernel_width) / stride_width + 1;
        int64_t w_row_end = std::min(w_im / stride_width + 1, width_row);
        int64_t h_row_start = (h_im < kernel_height)
            ? 0
            : (h_im - kernel_height) / stride_height + 1;
        int64_t h_row_end = std::min(h_im / stride_height + 1, height_row);

        scalar_t val = 0;
        for (int64_t h_row = h_row_start; h_row < h_row_end; h_row += 1) {
            for (int64_t w_row = w_row_start; w_row < w_row_end; w_row += 1) {
                int64_t h_k = (h_im - h_row * stride_height); // 元素在该[kw,kh]范围中所在行
                int64_t w_k = (w_im - w_row * stride_width); // 元素在该[kw,kh]范围中所在列
                // int64_t data_row_index = ((((batch_id * channels + h_row) * width_row +
                //                             w_row) * channels + c_im) * kernel_height +
                //                             h_k) * kernel_width + w_k;  // bug !!!!!!!!!!!!!!!!! -- fixed
                int64_t data_row_index = ((((batch_id * height_row + h_row) * width_row + w_row) * channels + c_im) * kernel_height + h_k) * kernel_width + w_k;
                val += data_row[data_row_index];
            }
        }
        data_im[((batch_id * channels + c_im) * height + h_im) * width + w_im] = val; // add this line
    }
}

void row2im_batch_cuda(
    cudaStream_t& stream,
    const at::Tensor& gradInput_rows,
    at::Tensor& gradInput,
    const int64_t kernel_height,
    const int64_t kernel_width,
    const int64_t pad_height,
    const int64_t pad_width,
    const int64_t stride_height,
    const int64_t stride_width)
{

    int64_t batch_size = gradInput.size(0);
    int64_t n_input_plane = gradInput.size(1);
    int64_t input_height = gradInput.size(2);
    int64_t input_width = gradInput.size(3);

    int64_t output_height = (input_height + 2 * pad_height - kernel_height) / stride_height + 1;
    int64_t output_width = (input_width + 2 * pad_width - kernel_width) / stride_width + 1;

    int64_t num_kernels = batch_size * n_input_plane * input_height * input_width;

    AT_DISPATCH_FLOATING_TYPES(gradInput_rows.scalar_type(), "row2im_batch_cuda", ([&] {
        row2im_batch_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
                num_kernels,
                gradInput_rows.data_ptr<scalar_t>(),
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
                gradInput.data_ptr<scalar_t>());
    }));
    MY_CUDA_CHECK(cudaGetLastError());
}

// for matrix_id, bucket_id in n_matrices * total_buckets:
//   bucket_index = buckets_index[matrix_id][bucket_id]
//   if bucket_index >= 0:
//   for i in H:
//     power[matrix_id][bucket_index][i] = 2^(H - i - 1) / (bucket_id + 1.0)
template <typename scalar_t>
__global__ void get_Power_kernel(
    const int64_t n,
    const int* __restrict__ buckets_index,
    scalar_t* __restrict__ power,
    const int64_t max_buckets,
    const int64_t total_buckets,
    const int64_t H)
{
    CUDA_1D_KERNEL_LOOP(index, n)
    {
        int64_t matrix_id = index / total_buckets;
        scalar_t bucket = (scalar_t)(index % total_buckets);
        int64_t bucket_id = buckets_index[index];
        if (bucket_id < 0)
            return;
        for (int64_t i = 0; i < H; i++) {
            power[matrix_id * max_buckets * H + bucket_id * H + i] = (1 << (H - 1 - i)) / (bucket + 1.0);
        }
    }
}

void get_Power(
    cudaStream_t& stream,
    const at::Tensor& buckets_index,
    at::Tensor& power,
    const int64_t max_buckets,
    const int64_t param_H)
{

    int64_t n_matrices = buckets_index.size(0);
    int64_t total_buckets = buckets_index.size(1);

    int64_t num_kernels = n_matrices * total_buckets;
    AT_DISPATCH_FLOATING_TYPES(power.scalar_type(), "get_Power", ([&] {
        get_Power_kernel<scalar_t>
            <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
                num_kernels,
                buckets_index.data_ptr<int>(),
                power.data_ptr<scalar_t>(),
                max_buckets,
                total_buckets,
                param_H);
    }));
    MY_CUDA_CHECK(cudaGetLastError());
}

template <typename scalar_t>
__global__ void get_gradHashValue_cuda_kernel(
    const int* __restrict__ vector_ids,
    const int* __restrict__ buckets_index_inv,
    const scalar_t* __restrict__ gradIndex,
    const int* __restrict__ buckets_count,
    scalar_t* __restrict__ grad_Hash_value,
    int n_matrices,
    int num_rows,
    int max_buckets,
    double sigma)
{
    CUDA_1D_KERNEL_LOOP(index, n_matrices * num_rows * max_buckets)
    {
        uint32_t bucket_id = index % max_buckets;
        index = index / max_buckets;
        uint32_t row_id = index % num_rows;
        uint32_t matrix_id = index / num_rows;

        uint32_t vector_id = vector_ids[matrix_id * num_rows + row_id] + 1;
        uint32_t bucket_index = buckets_index_inv[matrix_id * max_buckets + bucket_id] + 1;
        scalar_t grad_index = gradIndex[index];
        uint32_t bucket_count = buckets_count[matrix_id * max_buckets + bucket_id];

        scalar_t grad_Hash_value_temp = static_cast<scalar_t>(vector_id) / static_cast<scalar_t>(bucket_index) - 1;
        scalar_t neg_sigma_sq = -sigma * sigma;

        scalar_t grad_Hash_val = grad_Hash_value_temp / neg_sigma_sq;
        grad_Hash_val *= __expf(grad_Hash_value_temp * grad_Hash_val / 2);
        grad_Hash_val *= grad_index;
        grad_Hash_val /= bucket_count;

        grad_Hash_value[index] = __isnanf(grad_Hash_val) ? 0 : grad_Hash_val;
    }
}

void get_gradHashValue_cuda(
    cudaStream_t& stream,
    const at::Tensor& vector_ids,
    const at::Tensor& buckets_index_inv,
    const at::Tensor& gradIndex,
    const at::Tensor& buckets_count,
    at::Tensor& grad_Hash_value,
    int n_matrices,
    int num_rows,
    int max_buckets,
    double sigma)
{
    AT_DISPATCH_FLOATING_TYPES(gradIndex.scalar_type(), "get_gradHashValue", ([&] {
        get_gradHashValue_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(num_rows * n_matrices * max_buckets), CUDA_NUM_THREADS, 0, stream>>>(
                vector_ids.data_ptr<int>(),
                buckets_index_inv.data_ptr<int>(),
                gradIndex.data_ptr<scalar_t>(),
                buckets_count.data_ptr<int>(),
                grad_Hash_value.data_ptr<scalar_t>(),
                n_matrices,
                num_rows,
                max_buckets,
                sigma);
    }));
    MY_CUDA_CHECK(cudaGetLastError());
}
