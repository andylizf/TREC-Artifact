import torch
from trec.conv_layer import Conv2d_TREC
import time

def profile_conv_trec():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 设置输入参数
    batch_size = 64
    in_channels = 48
    out_channels = 12
    input_size = 32
    kernel_size = 3
    
    # 创建输入张量
    inputs = torch.randn(batch_size, in_channels, input_size, input_size).to(device)
    
    # 创建 Conv2d_TREC 层
    conv_trec = Conv2d_TREC(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        param_L=12,  # 可以根据需要调整
        param_H=8,  # 可以根据需要调整
        layer=0,    # 层索引
        padding=1,  # 保持输出大小不变
        stride=1
    ).to(device)
    
    # 预热 GPU
    for _ in range(10):
        _ = conv_trec(inputs)
    torch.cuda.synchronize()
    
    # 计时运行
    num_runs = 100
    start_time = time.time()
    
    for _ in range(num_runs):
        output = conv_trec(inputs)
        torch.cuda.synchronize()
    
    end_time = time.time()
    avg_time = (end_time - start_time) / num_runs
    
    # 打印结果
    print(f"Input shape: {inputs.shape}")
    print(f"Weight shape: {conv_trec.weight.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Average time per forward pass: {avg_time*1000:.2f} ms")
    
    return conv_trec, inputs

if __name__ == "__main__":
    conv_trec, inputs = profile_conv_trec()
    
    # 保存输入和模型参数以供后续分析
    torch.save({
        'inputs': inputs.cpu(),
        'weight': conv_trec.weight.cpu(),
        'bias': conv_trec.bias.cpu() if conv_trec.bias is not None else None,
        'random_vectors': conv_trec.random_vectors.cpu(),
    }, 'conv_trec_profile_data.pt') 