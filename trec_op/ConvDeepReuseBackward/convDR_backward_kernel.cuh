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
    const int* vector_index,
    const scalar_t* gradOutput_mat, // {batch_size * outputHeight * outputWidth, nOutputPlane}
    scalar_t* gradOutput_centroids, // {n_matrices, max_buckets, nOutputPlane}
    const int64_t max_buckets,
    const int64_t n_output_plane,
    const int64_t num_rows, // batch_size * outputHeight * outputWidth
    const int64_t n_matrices)
{
    // printf("get_gradOutputCentroids_add_cuda_kernel\n");
    CUDA_1D_KERNEL_LOOP(global_id, num_rows * n_matrices)
    {
        int vect_index = vector_index[global_id];
        // int64_t matrix_offset = global_id % num_rows;
        int64_t matrix_id = global_id / num_rows;
        int64_t vec_id = global_id % num_rows;
        int64_t buck_offset = matrix_id * max_buckets + vect_index;

        scalar_t* buck_start = gradOutput_centroids + buck_offset * n_output_plane;
        // const scalar_t *vect_start = gradOutput_mat + global_id * n_output_plane;   // ??? global_id, sure?
        const scalar_t* vect_start = gradOutput_mat + vec_id * n_output_plane;
        for (int i = 0; i < n_output_plane; i++) {
            // printf("block (%d, %d, %d) thread(%d, %d, %d)\ngradOutput_mat[%ld][%d][%d] = %f\n",
            //     blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z,
            //     matrix_id, vect_index, i, vect_start[i]);
            atomicAdd(buck_start + i, vect_start[i]);
        }
        // for all matrix_id in n_matrices:
        // for all i in n_output_plane:
        //  gradOutput_centroids[matrix_id][vect_index[matrix_id][vec_id]][i] += gradOutput_mat[vec_id][i]
    }
}

void get_gradOutputCentroids_add_cuda(
    cudaStream_t& stream,
    const at::Tensor& vector_index,
    const at::Tensor& gradOutput_mat,
    at::Tensor& gradOutput_centroids)
{
    printf("get_gradOutputCentroids_add_cuda\n");
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
    // ! fixme: cuda memory error
    MY_CUDA_CHECK(cudaGetLastError());
}

template <typename scalar_t>
__global__ void get_gradOutputCentroids_div_cuda_kernel(
    scalar_t* gradOutput_centroids,
    const int* buckets_count,
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
    printf("get_gradOutputCentroids_div_cuda\n");
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

template <typename scalar_t>
__global__ void reconstruct_gradInputRows_cuda_kernel(
    const int* vector_index,
    const scalar_t* gradInput_centroids,
    scalar_t* gradInput_rows,
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

        scalar_t* row_start = gradInput_rows + (matrix_offset * n_matrices + matrix_id) * param_L;
        const scalar_t* buck_start = gradInput_centroids + (matrix_id * max_buckets + vect_index) * param_L;

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
    const scalar_t* data_row,
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
    scalar_t* data_im)
{

    CUDA_1D_KERNEL_LOOP(index, n)
    {
        int64_t w_im = index % width + pad_width;
        int64_t idx = index / width;
        int64_t h_im = idx % height + pad_height;
        idx = idx / height;
        int64_t c_im = idx % channels;
        int64_t batch_id = idx / channels;

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

template <typename scalar_t>
__global__ void get_Power_kernel(
    const int64_t n,
    const int* buckets_index,
    // const int* buckets_count,
    // const int* vector_index,
    // scalar_t* gradIndex,
    // scalar_t* grad_Hash_value,
    scalar_t* power,
    // const int64_t n_matrices,
    const int64_t max_buckets,
    const int64_t total_buckets,
    // const int64_t num_rows,
    const int64_t H
    // const float sigma
)
{

    CUDA_1D_KERNEL_LOOP(index, n)
    {
        // int64_t idx = index / num_rows;
        int64_t matrix_id = index / total_buckets;
        scalar_t bucket = (scalar_t)(index % total_buckets);
        // printf("1\n");
        int64_t bucket_id = buckets_index[index];
        // printf("2\n");
        if (bucket_id < 0)
            return;
        // int64_t count = buckets_count[matrix_id * max_buckets + bucket_id];

        // for (int64_t vid = 0; vid < num_rows; vid++) {
        //     // gradIndex[matrix_id * max_buckets * num_rows + bucket_id * num_rows + vid] /= count;
        //     scalar_t value = vector_index[matrix_id * num_rows + vid] / (bucket + 1.0) - 1;
        //     // if(bucket == 598 && n_matrices==0)
        //     //     printf("numer=%f, denomer=%f, value=%f\n", Hash_Value[matrix_id * max_buckets * num_rows + bucket_id * num_rows + vid], bucket + 1.0, value);
        //     grad_Hash_value[matrix_id * max_buckets * num_rows + vid * max_buckets + bucket_id] = value;
        //     gradIndex[matrix_id * max_buckets * num_rows + vid * max_buckets + bucket_id] /= count;
        //     // grad_Hash_value[matrix_id * max_buckets * num_rows + vid * max_buckets + bucket_id] = -1 * value / (sigma * sigma)
        //     //                 * exp(-1 * value * value / (2 * sigma * sigma))
        //     //                 * gradIndex[matrix_id * max_buckets * num_rows + bucket_id * num_rows + vid] / count;
        //     // grad_Hash_value[matrix_id * max_buckets * num_rows + bucket_id * num_rows + vid] = value;
        //     // printf("value=%f, value2=%f, gaussian=%f, gradHash=%f\n", value, -1 * (value - 1) / (sigma * sigma), exp(-1 * (value - 1) * (value - 1) / (2 * sigma * sigma)), gradIndex[matrix_id * max_buckets * num_rows + bucket_id * num_rows + vid] / count);
        // }
        for (int64_t i = 0; i < H; i++) {
            power[matrix_id * max_buckets * H + bucket_id * H + i] = pow(2, H - 1 - i) / (bucket + 1.0);
        }
    }
}

void get_Power(
    cudaStream_t& stream,
    const at::Tensor& buckets_index,
    // const at::Tensor &buckets_count,
    // const at::Tensor &vector_index,
    // at::Tensor &gradIndex,
    // at::Tensor &grad_Hash_value,
    at::Tensor& power,
    const int64_t max_buckets,
    const int64_t param_H)
{

    int64_t n_matrices = buckets_index.size(0);
    // int64_t max_buckets = buckets_count.size(1);
    // int64_t num_rows = vector_index.size(1);
    // std::cout << n_matrices << ", " << max_buckets << ", " << num_rows << std::endl;
    int64_t total_buckets = buckets_index.size(1);
    // std::cout << n_matrices << ", " << max_buckets << ", " << num_rows << ", " << total_buckets << std::endl;

    int64_t num_kernels = n_matrices * total_buckets;
    AT_DISPATCH_FLOATING_TYPES(power.scalar_type(), "get_Power", ([&] {
        get_Power_kernel<scalar_t>
            <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
                num_kernels,
                buckets_index.data_ptr<int>(),
                // buckets_count.data_ptr<int>(),
                // vector_index.data_ptr<int>(),
                // gradIndex.data_ptr<scalar_t>(),
                // grad_Hash_value.data_ptr<scalar_t>(),
                power.data_ptr<scalar_t>(),
                // n_matrices,
                max_buckets,
                total_buckets,
                // num_rows,
                param_H);
    }));
    MY_CUDA_CHECK(cudaGetLastError());
}