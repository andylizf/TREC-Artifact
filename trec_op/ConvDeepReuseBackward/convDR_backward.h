#pragma once
#include <torch/extension.h>


std::vector<at::Tensor> conv_deep_reuse_backward(
        // const at::IntArrayRef input_size,
        const at::Tensor input_row,     // {n_matrices, num_row, param_L}
        const at::Tensor inputCentroids,  // {n_matrices, max_buckets, param_L}
        const at::Tensor weights,
        const at::Tensor gradOutput,      // {N, K, outH, outW}
        const at::Tensor vector_index,    // {n_matrices, num_rows}
        const at::Tensor vector_ids,    // {n_matrices, num_rows}
        const at::Tensor buckets_count,   // {n_matrices, max_buckets}
        const at::Tensor buckets_index,   // {n_matrices, total_buckets}
        const at::Tensor buckets_index_inv,   // {n_matrices, max_buckets}
        const at::Tensor random_vectors,  // {L, H}
        const int64_t input_height,
        const int64_t input_width,
        const int64_t pad_height,
        const int64_t pad_width,
        const int64_t stride_height,
        const int64_t stride_width,
        const int64_t param_H,
        const float alpha,
        const float sigma,
        const bool do_bias);