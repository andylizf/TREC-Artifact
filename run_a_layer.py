import torch
from trec.conv_layer import Conv2d_TREC
import time
import argparse
import sys
import os
from contextlib import contextmanager

@contextmanager
def suppress_output():
    """临时禁用标准输出和标准错误"""
    # 保存当前的文件描述符
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    
    # 复制文件描述符
    stdout_dup = os.dup(stdout_fd)
    stderr_dup = os.dup(stderr_fd)
    
    try:
        # 打开空设备
        devnull = open(os.devnull, 'w')
        devnull_fd = devnull.fileno()
        
        # 重定向标准输出和标准错误到空设备
        os.dup2(devnull_fd, stdout_fd)
        os.dup2(devnull_fd, stderr_fd)
        
        yield
    finally:
        # 恢复原始的文件描述符
        os.dup2(stdout_dup, stdout_fd)
        os.dup2(stderr_dup, stderr_fd)
        
        # 关闭所有复制的文件描述符
        os.close(stdout_dup)
        os.close(stderr_dup)
        devnull.close()

def profile_conv_trec(batch_size=256, in_channels=48, out_channels=12, 
                     input_size=32, kernel_size=3, param_L=12, param_H=8,
                     num_warmup=10, num_runs=100, do_backward=False, save_data=False):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建输入张量
    inputs = torch.randn(batch_size, in_channels, input_size, input_size).to(device)
    if do_backward:
        inputs.requires_grad_()
    
    # 创建 Conv2d_TREC 层
    conv_trec = Conv2d_TREC(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        param_L=param_L,
        param_H=param_H,
        layer=0,
        padding=1,
        stride=1,
        bias=True
    ).to(device)
    
    # 预热 GPU (禁用所有输出)
    with suppress_output():
        for _ in range(num_warmup):
            _ = conv_trec(inputs)
            if do_backward:
                target = torch.randint(0, out_channels, (batch_size,)).to(device)
                criterion = torch.nn.CrossEntropyLoss()
                loss = criterion(_.mean(dim=(2, 3)), target)
                loss.backward()
            torch.cuda.synchronize()
    
    # 计时运行
    start_time = time.time()
    
    for _ in range(num_runs):
        output = conv_trec(inputs)
        if do_backward:
            target = torch.randint(0, out_channels, (batch_size,)).to(device)
            criterion = torch.nn.CrossEntropyLoss()
            loss = criterion(output.mean(dim=(2, 3)), target)
            loss.backward()
        torch.cuda.synchronize()
    
    end_time = time.time()
    avg_time = (end_time - start_time) / num_runs
    
    # 打印结果，使用不同的格式避免与时间数据混淆
    print("============ Configuration ============")
    print(f"BATCH_SIZE {batch_size}")
    print(f"IN_CHANNELS {in_channels}")
    print(f"OUT_CHANNELS {out_channels}")
    print(f"INPUT_SIZE {input_size}")
    print(f"KERNEL_SIZE {kernel_size}")
    print(f"PARAM_L {param_L}")
    print(f"PARAM_H {param_H}")
    print(f"DO_BACKWARD {do_backward}")
    print("=====================================")
    
    print("============ Shape Info ============")
    print(f"INPUT_SHAPE {list(inputs.shape)}")
    print(f"WEIGHT_SHAPE {list(conv_trec.weight.shape)}")
    print(f"OUTPUT_SHAPE {list(output.shape)}")
    print("===================================")
    
    if do_backward:
        print("============ Gradient Info ============")
        print(f"INPUTS_GRAD_NAN {torch.isnan(inputs.grad).any()}")
        print(f"WEIGHT_GRAD_NAN {torch.isnan(conv_trec.weight.grad).any()}")
        print(f"BIAS_GRAD_NAN {torch.isnan(conv_trec.bias.grad).any()}")
        print(f"RANDOM_VECTORS_GRAD_NAN {torch.isnan(conv_trec.random_vectors.grad).any()}")
        print("=====================================")
    
    if save_data:
        torch.save({
            'inputs': inputs.cpu(),
            'weight': conv_trec.weight.cpu(),
            'bias': conv_trec.bias.cpu() if conv_trec.bias is not None else None,
            'random_vectors': conv_trec.random_vectors.cpu(),
        }, 'conv_trec_profile_data.pt')
    
    return conv_trec, inputs, output

def profile_baseline(batch_size=64, in_channels=48, out_channels=12, 
                    input_size=32, kernel_size=3, num_warmup=10, num_runs=100):
    """运行普通Conv2d作为基准"""
    # 禁用 cudnn
    torch.backends.cudnn.enabled = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建输入张量
    inputs = torch.randn(batch_size, in_channels, input_size, input_size).to(device)
    
    # 创建普通Conv2d层
    conv = torch.nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        padding=1,
        stride=1,
        bias=True
    ).to(device)
    
    # 预热
    with suppress_output():
        for _ in range(num_warmup):
            _ = conv(inputs)
            torch.cuda.synchronize()
    
    # 计时运行
    times = []
    for i in range(num_runs):
        torch.cuda.synchronize()
        start = time.time()
        
        output = conv(inputs)
        
        torch.cuda.synchronize()
        end = time.time()
        elapsed = end - start
        times.append(elapsed)
        print(f"Baseline forward pass: {elapsed}")
    
    avg_time = sum(times) / len(times)
    std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
    
    # 打印配置信息
    print("============ Configuration ============")
    print(f"BATCH_SIZE {batch_size}")
    print(f"IN_CHANNELS {in_channels}")
    print(f"OUT_CHANNELS {out_channels}")
    print(f"INPUT_SIZE {input_size}")
    print(f"KERNEL_SIZE {kernel_size}")
    print(f"CUDNN_ENABLED False")
    print("=====================================")
    
    print("============ Shape Info ============")
    print(f"INPUT_SHAPE {list(inputs.shape)}")
    print(f"WEIGHT_SHAPE {list(conv.weight.shape)}")
    print(f"OUTPUT_SHAPE {list(output.shape)}")
    print("===================================")
    
    print("============ Timing Info ============")
    print(f"AVERAGE TIME {avg_time:.6f} ± {std_time:.6f} seconds")
    print(f"MIN TIME: {min(times):.6f} seconds")
    print(f"MAX TIME: {max(times):.6f} seconds")
    print("====================================")
    
    # 恢复 cudnn 设置
    torch.backends.cudnn.enabled = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Profile Conv2d_TREC layer')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--in-channels', type=int, default=48)
    parser.add_argument('--out-channels', type=int, default=12)
    parser.add_argument('--input-size', type=int, default=32)
    parser.add_argument('--kernel-size', type=int, default=3)
    parser.add_argument('--param-L', type=int, default=12)
    parser.add_argument('--param-H', type=int, default=8)
    parser.add_argument('--num-warmup', type=int, default=10)
    parser.add_argument('--num-runs', type=int, default=100)
    parser.add_argument('--do-backward', action='store_true')
    parser.add_argument('--save-data', action='store_true')
    parser.add_argument('--baseline', action='store_true', 
                       help='Run baseline Conv2d instead of TREC')
    
    args = parser.parse_args()
    
    if args.baseline:
        profile_baseline(
            batch_size=args.batch_size,
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            input_size=args.input_size,
            kernel_size=args.kernel_size,
            num_warmup=args.num_warmup,
            num_runs=args.num_runs
        )
    else:
        conv_trec, inputs, output = profile_conv_trec(
            batch_size=args.batch_size,
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            input_size=args.input_size,
            kernel_size=args.kernel_size,
            param_L=args.param_L,
            param_H=args.param_H,
            num_warmup=args.num_warmup,
            num_runs=args.num_runs,
            do_backward=args.do_backward,
            save_data=args.save_data
        )
