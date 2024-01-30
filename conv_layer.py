from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import collections.abc
from itertools import repeat
import trec as _C
import save_tensor as st


print_rc = False

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_pair = _ntuple(2)


class Conv2d_TREC_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weights, bias, random_vectors, stride, padding, 
                param_L, param_H, is_training, layer, sigma, alpha, do_bias=True):
        outputs = _C.conv_deep_reuse_forward(inputs, weights, bias, random_vectors,
                                            padding[0], padding[1], stride[0], stride[1], 
                                            param_L, param_H, do_bias, is_training, print_rc)

        if is_training:
            _, inputCentroids, vector_index, vector_ids, buckets_count, buckets_index, buckets_index_inv, input_row  = outputs
            variables = [input_row, inputCentroids, vector_index, vector_ids, buckets_count, buckets_index, buckets_index_inv, random_vectors, weights]
            ctx.save_for_backward(*variables)
            ctx.mark_non_differentiable(inputCentroids, vector_index, vector_ids, buckets_count, buckets_index, buckets_index_inv)
            ctx.stride = stride
            ctx.padding = padding
            ctx.layer = layer
            ctx.H = param_H
            ctx.alpha = alpha
            ctx.sigma = sigma
            ctx.do_bias = do_bias
            ctx.input_height = inputs.size()[2]
            ctx.input_width = inputs.size()[3]
        return outputs[0]

    @staticmethod
    def backward(ctx, gradOutput):
        input_row, inputCentroids, vector_index, vector_ids, buckets_count, buckets_index, buckets_index_inv, random_vectors, weights = ctx.saved_tensors
        stride_height, stride_width = ctx.stride
        padding_height, padding_width = ctx.padding
        grads = _C.conv_deep_reuse_backward(input_row, inputCentroids, weights, 
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
            in_channels, out_channels, kernel_size, stride, padding, dilation=_pair(1),
            transposed=False, output_padding=_pair(0), groups=1, bias=True, padding_mode='zeros')
        
        self.random_vectors = nn.Parameter(torch.randn((param_L, param_H)).cuda())

    def forward(self, inputs):
        return Conv2d_TREC_Function.apply(inputs, self.weight, self.bias, self.random_vectors, self.stride, self.padding,
                                self.param_L, self.param_H, self.training, self.layer, self.sigma, self.alpha, self.do_bias)
