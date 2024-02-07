#pragma once
#include <torch/extension.h>

std::vector<at::Tensor> conv_deep_reuse_forward(const at::Tensor& input,
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
                                                const bool is_training,
                                                const bool print_rc);

// at::Tensor slow_conv2d_forward(
//            const at::Tensor &input,
//            const at::Tensor &weight_,
//            const at::Tensor &bias,
//            int64_t kH, int64_t kW,
//            int64_t dH, int64_t dW,
//            int64_t padH, int64_t padW);
// std::vector<at::Tensor> conv_native_forward(at::Tensor input,
//                                             at::Tensor weights,
//                                             at::Tensor bias,
//                                             int64_t pad_height,
//                                             int64_t pad_width,
//                                             int64_t stride_height,
//                                             int64_t stride_width);