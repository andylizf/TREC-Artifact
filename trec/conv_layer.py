import collections.abc
from itertools import repeat

import torch
import torch.nn as nn

from ._C import conv_deep_reuse_backward, conv_deep_reuse_forward

__all__ = ["Conv2d_TREC"]


def _ntuple(n, name="parse"):
    # Modified from torch.nn.modules.utils

    def parse(x) -> tuple:
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    parse.__name__ = name
    return parse


_pair = _ntuple(2, "_pair")


class Conv2d_TREC_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weights, bias, random_vectors, stride, padding,
                param_L, param_H, is_training, layer, sigma, alpha, do_bias=True):
        # Print settings for each layer
        print(f"\nLayer {layer} Settings:")
        print(f"Input shape: {inputs.shape}")
        print(f"Weight shape: {weights.shape}")
        print(f"Stride: {stride}, Padding: {padding}")
        print(f"param_L: {param_L}, param_H: {param_H}")
        print(f"sigma: {sigma:.6f}, alpha: {alpha}")
        print(f"Training mode: {is_training}, Use bias: {do_bias}")

        timer_start = torch.cuda.Event(enable_timing=True)
        timer_end = torch.cuda.Event(enable_timing=True)
        timer_start.record()  # type: ignore
        
        outputs = conv_deep_reuse_forward(inputs, weights, bias, random_vectors,
                                          padding[0], padding[1], stride[0], stride[1],
                                          param_L, param_H, do_bias, is_training)

        timer_end.record()  # type: ignore
        torch.cuda.synchronize()
        print(f"Conv2d_TREC time: {timer_start.elapsed_time(timer_end)}")

        if is_training:
            _, inputCentroids, vector_index, vector_ids, buckets_count, buckets_index, buckets_index_inv, input_row = outputs
            variables = [input_row, inputCentroids, vector_index, vector_ids,
                         buckets_count, buckets_index, buckets_index_inv, random_vectors, weights]
            ctx.save_for_backward(*variables)
            ctx.mark_non_differentiable(
                inputCentroids, vector_index, vector_ids, buckets_count, buckets_index, buckets_index_inv)
            ctx.stride = stride
            ctx.padding = padding
            ctx.layer = layer
            ctx.H = param_H
            ctx.alpha = alpha
            ctx.sigma = sigma
            ctx.do_bias = do_bias
            ctx.input_height = inputs.size()[2]
            ctx.input_width = inputs.size()[3]
        return outputs[0]  # used for gradient computation

    @staticmethod
    def backward(ctx, gradOutput):
        input_row, inputCentroids, vector_index, vector_ids, buckets_count, buckets_index, buckets_index_inv, random_vectors, weights = ctx.saved_tensors
        stride_height, stride_width = ctx.stride
        padding_height, padding_width = ctx.padding
        grads = conv_deep_reuse_backward(input_row, inputCentroids, weights,
                                         gradOutput, vector_index, vector_ids, buckets_count,
                                         buckets_index, buckets_index_inv, random_vectors,
                                         ctx.input_height, ctx.input_width,
                                         padding_height, padding_width,
                                         stride_height, stride_width,
                                         ctx.H, ctx.alpha, ctx.sigma, ctx.do_bias)

        if ctx.do_bias:
            gradInput, gradWeight, gradBias, gradHash2 = grads
        else:
            gradInput, gradWeight, gradHash2 = grads
            gradBias = None
        return gradInput, gradWeight, gradBias, gradHash2, None, None, None, None, None, None, None, None, None, None, None


class Conv2d_TREC(nn.modules.conv._ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size,
                 param_L, param_H, layer, padding=0, stride=1, groups=1,
                 alpha=10000, k=5.0, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        self.param_L = param_L
        self.param_H = param_H
        self.layer = layer
        self.sigma = 1.0 / pow(2, param_H) / k
        self.alpha = alpha
        self.do_bias = bias

        super(Conv2d_TREC, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation=_pair(
                1),
            transposed=False, output_padding=_pair(0), groups=1, bias=True, padding_mode='zeros')

        self.random_vectors = nn.Parameter(
            torch.randn((param_L, param_H)).cuda())

    def forward(self, inputs):
        return Conv2d_TREC_Function.apply(inputs, self.weight, self.bias, self.random_vectors, self.stride, self.padding,
                                          self.param_L, self.param_H, self.training, self.layer, self.sigma, self.alpha, self.do_bias)
