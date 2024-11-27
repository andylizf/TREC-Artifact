import torch
from trec.conv_layer import Conv2d_TREC
import time

def profile_conv_trec():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 设置输入参数
    batch_size = 64
    in_channels = 48
    out_channels = 12
    input_size = 32
    kernel_size = 3
    
    print(f"\nNetwork parameters:")
    print(f"batch_size: {batch_size}")
    print(f"in_channels: {in_channels}")
    print(f"out_channels: {out_channels}")
    print(f"input_size: {input_size}")
    print(f"kernel_size: {kernel_size}")
    
    # 创建输入张量
    inputs = torch.randn(batch_size, in_channels, input_size, input_size).to(device)
    print(f"\nInput tensor shape: {inputs.shape}")
    
    # 创建 Conv2d_TREC 层
    conv_trec = Conv2d_TREC(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        param_L=12,
        param_H=8,
        layer=0,
        padding=1,
        stride=1
    ).to(device)
    
    # 设置为评估模式
    conv_trec.eval()
    
    print(f"\nConv2d_TREC parameters:")
    print(f"param_L: {conv_trec.param_L}")
    print(f"param_H: {conv_trec.param_H}")
    print(f"Training mode: {conv_trec.training}")
    
    # 预热 GPU
    print("\nWarming up GPU...")
    for i in range(10):
        _ = conv_trec(inputs)
        if i == 0:
            print(f"First forward pass completed")
    torch.cuda.synchronize()
    print("Warm-up completed")
    
    # 计时运行
    num_runs = 100
    print(f"\nStarting {num_runs} timed runs...")
    start_time = time.time()
    
    for i in range(num_runs):
        output = conv_trec(inputs)
        if i == 0:
            print(f"Output tensor shape: {output.shape}")
            print(f"Number of returned tensors: {len(output) if isinstance(output, (list, tuple)) else 1}")
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