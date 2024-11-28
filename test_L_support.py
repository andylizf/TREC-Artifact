import torch
import numpy as np
from trec.conv_layer import conv_deep_reuse_forward

# 创建一个不能被L整除的情况
in_channels = 3
kernel_size = 3  # 3x3 kernel
param_L = 7  # 不能整除 in_channels * kernel_size * kernel_size (=27)

# 创建输入数据
batch_size = 2
height = 32
width = 32
inputs = torch.randn(batch_size, in_channels, height, width).cuda()

# 创建权重
out_channels = 16
weights = torch.randn(out_channels, in_channels, kernel_size, kernel_size).cuda()

# 创建偏置
bias = torch.randn(out_channels).cuda()

# 创建随机向量
param_H = 8
random_vectors = torch.randn(param_L, param_H).cuda()

# 运行测试
output = conv_deep_reuse_forward(
    inputs,          # arg0: input tensor 
    weights,         # arg1: weight tensor
    bias,           # arg2: bias tensor
    random_vectors,  # arg3: random vectors tensor
    1,              # arg4: pad_height
    1,              # arg5: pad_width
    1,              # arg6: stride_height
    1,              # arg7: stride_width
    param_L,        # arg8: param_L
    param_H,        # arg9: param_H
    True,           # arg10: do_bias
    True            # arg11: is_training
)

# 打印形状和其他信息进行验证
print(f"Input shape: {inputs.shape}")
print(f"Output shape: {output[0].shape}")  # 训练模式下返回多个tensor，第一个是输出
print(f"Row length (in_channels * kernel_size^2): {in_channels * kernel_size * kernel_size}")
print(f"Param L: {param_L}")
print(f"Number of matrices: {(in_channels * kernel_size * kernel_size + param_L - 1) // param_L}")

# 如果在训练模式下，打印返回的其他tensor的形状
if len(output) > 1:
    print("\nTraining mode outputs:")
    for i, tensor in enumerate(output):
        print(f"Output[{i}] shape: {tensor.shape}")