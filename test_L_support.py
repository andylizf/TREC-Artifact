import torch
import numpy as np
from trec.conv_layer import Conv2d_TREC

# 创建一个不能被L整除的情况
in_channels = 3
kernel_size = 3  # 3x3 kernel
param_L = 9  # 不能整除 in_channels * kernel_size * kernel_size (=27)

# 创建输入数据
batch_size = 2
height = 32
width = 32
inputs = torch.randn(batch_size, in_channels, height, width).cuda()
inputs.requires_grad_()

# 创建 Conv2d_TREC 层
conv_layer = Conv2d_TREC(
    in_channels=in_channels,
    out_channels=16,
    kernel_size=kernel_size,
    param_L=param_L,
    param_H=8,
    layer=0,
    padding=1,
    stride=1,
    bias=True
).cuda()

# 运行前向传播
output = conv_layer(inputs)
print("Output contiguous:", output.is_contiguous())

# 创建模拟标签和损失函数
target = torch.randint(0, 16, (batch_size,)).cuda()  # 随机标签
criterion = torch.nn.CrossEntropyLoss()

# 计算损失并反向传播
loss = criterion(output.mean(dim=(2, 3)), target)  # 对空间维度取平均，模拟全局平均池化
print("Before backward")
loss.backward()
print("After backward")

# 验证形状
print(f"Input shape: {inputs.shape}")
print(f"Output shape: {output.shape}")
print(f"Row length (in_channels * kernel_size^2): {in_channels * kernel_size * kernel_size}")
print(f"Param L: {param_L}")
print(f"Number of matrices: {(in_channels * kernel_size * kernel_size + param_L - 1) // param_L}")

# 验证梯度
print("\nGradient shapes:")
print(f"inputs.grad shape: {inputs.grad.shape}")
print(f"conv_layer.weight.grad shape: {conv_layer.weight.grad.shape}")
print(f"conv_layer.bias.grad shape: {conv_layer.bias.grad.shape}")
print(f"conv_layer.random_vectors.grad shape: {conv_layer.random_vectors.grad.shape}")

# 验证梯度是否包含 NaN
print("\nGradient checks:")
print(f"inputs.grad contains NaN: {torch.isnan(inputs.grad).any()}")
print(f"conv_layer.weight.grad contains NaN: {torch.isnan(conv_layer.weight.grad).any()}")
print(f"conv_layer.bias.grad contains NaN: {torch.isnan(conv_layer.bias.grad).any()}")
print(f"conv_layer.random_vectors.grad contains NaN: {torch.isnan(conv_layer.random_vectors.grad).any()}")
