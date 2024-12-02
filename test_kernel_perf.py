import time
from dataclasses import dataclass

import cutlass
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sympy import divisors

from trec.conv_layer import Conv2d_TREC
from trec_classic.conv_layer import Conv2d_TREC as Conv2d_TREC_Classic


@dataclass
class ConvConfig:
    batch_size: int
    input_channels: int
    height: int
    width: int
    output_channels: int
    kernel_size: int


def generate_data(config: ConvConfig):
    batch_size = config.batch_size
    input_channels = config.input_channels
    height = config.height
    width = config.width
    output_channels = config.output_channels
    kernel_size = config.kernel_size

    input_data = np.random.randn(
        batch_size, input_channels, height, width).astype(np.float32)
    kernel_data = np.random.randn(
        output_channels, input_channels, kernel_size, kernel_size).astype(np.float32)
    return torch.tensor(input_data), torch.tensor(kernel_data)


torch.manual_seed(0)
np.random.seed(0)

TRAINING = False
TIMES = 10

batch_size = 1 if not TRAINING else 32
configs = [
    # ConvConfig(batch_size, 1, 1, 1, 1, 1),

    # # Cifar10
    # ConvConfig(batch_size, 64, 14, 14, 64, 5),

    # # ImageNet
    # ConvConfig(batch_size, 3, 224, 224, 64, 7),
    # ConvConfig(batch_size, 64, 56, 56, 128, 3),
    # ConvConfig(batch_size, 128, 28, 28, 256, 3),
    # ConvConfig(batch_size, 256, 14, 14, 512, 3),

    # # VGGNet
    # ConvConfig(batch_size, 3, 224, 224, 64, 3),
    # ConvConfig(batch_size, 64, 112, 112, 128, 3),
    # ConvConfig(batch_size, 128, 56, 56, 256, 3),
    # ConvConfig(batch_size, 256, 28, 28, 512, 3),

    # # ResNet
    # ConvConfig(batch_size, 3, 224, 224, 64, 7),
    # ConvConfig(batch_size, 64, 56, 56, 128, 3),
    # ConvConfig(batch_size, 128, 28, 28, 256, 3),
    # ConvConfig(batch_size, 256, 14, 14, 512, 3),

    # # MobileNet
    # ConvConfig(batch_size, 3, 224, 224, 32, 3),
    # ConvConfig(batch_size, 32, 112, 112, 64, 3),
    # ConvConfig(batch_size, 64, 56, 56, 128, 3),
    # ConvConfig(batch_size, 128, 28, 28, 256, 3),

    # # CIFAR-10
    # ConvConfig(batch_size, 3, 32, 32, 64, 5),
    # ConvConfig(batch_size, 64, 16, 16, 128, 5),
    # ConvConfig(batch_size, 128, 8, 8, 256, 5),

    # # 基础配置
    # ConvConfig(batch_size, 3, 1, 1, 1, 1),
    # ConvConfig(batch_size, 3, 32, 32, 64, 3),
    # ConvConfig(batch_size, 64, 16, 16, 128, 3),
    # ConvConfig(batch_size, 128, 8, 8, 256, 3),

    # # 基础图像处理
    # ConvConfig(batch_size, 3, 32, 32, 16, 3),

    # # VGG风格的早期层
    # ConvConfig(batch_size, 64, 224, 224, 64, 3),

    # # ResNet风格的中间层
    # ConvConfig(batch_size, 128, 56, 56, 128, 3),

    # # 深层网络的后期层
    # ConvConfig(batch_size, 512, 14, 14, 512, 3),

    # # 1x1卷积（用于通道数调整）
    # ConvConfig(batch_size, 256, 28, 28, 64, 1),

    # # 空洞卷积（常用于语义分割）
    # ConvConfig(batch_size, 256, 64, 64, 256, 3),

    # # 深度可分离卷积（MobileNet风格）
    # ConvConfig(batch_size, 32, 112, 112, 32, 3),
    # ConvConfig(batch_size, 32, 112, 112, 64, 1),

    # # 转置卷积（用于上采样）
    # ConvConfig(batch_size, 128, 32, 32, 64, 4),

    # # Stable Diffusion U-Net的下采样路径
    # ConvConfig(batch_size, 320, 64, 64, 320, 3),
    # ConvConfig(batch_size, 320, 64, 64, 640, 3),
    # ConvConfig(batch_size, 640, 32, 32, 640, 3),
    # ConvConfig(batch_size, 640, 32, 32, 1280, 3),
    # ConvConfig(batch_size, 1280, 16, 16, 1280, 3),

    # # Stable Diffusion U-Net的中间块
    # ConvConfig(batch_size, 1280, 8, 8, 1280, 3),

    # # Stable Diffusion U-Net的上采样路径
    # ConvConfig(batch_size, 1280, 16, 16, 1280, 3),
    # ConvConfig(batch_size, 1280, 16, 16, 640, 3),
    # ConvConfig(batch_size, 640, 32, 32, 640, 3),
    # ConvConfig(batch_size, 640, 32, 32, 320, 3),

    # # Stable Diffusion中的注意力块
    # ConvConfig(batch_size, 320, 64, 64, 320, 1),

    # # Stable Diffusion的输出层
    # ConvConfig(batch_size, 320, 64, 64, 4, 3),

    # # 条件嵌入处理
    # ConvConfig(batch_size, 768, 1, 1, 1280, 1),
    ConvConfig(batch_size, 7, 160, 832, 7, 160),
    ConvConfig(batch_size, 7, 256, 832, 7, 256),
    ConvConfig(batch_size, 7, 48, 832, 7, 48),
    # ConvConfig(batch_size, 13, 256, 64, 13, 256),
    ConvConfig(batch_size, 14, 112, 512, 14, 112),
    ConvConfig(batch_size, 14, 160, 512, 14, 160),
    ConvConfig(batch_size, 14, 192, 480, 14, 192),
    ConvConfig(batch_size, 14, 96, 480, 14, 96),
    ConvConfig(batch_size, 27, 48, 384, 27, 48),
    ConvConfig(batch_size, 28, 16, 192, 28, 16),
    ConvConfig(batch_size, 28, 64, 192, 28, 64),
    ConvConfig(batch_size, 55, 16, 128, 55, 16),
    # ConvConfig(batch_size, 55, 64, 16, 55, 64),
    ConvConfig(batch_size, 7, 160, 832, 7, 160),
    ConvConfig(batch_size, 7, 256, 832, 7, 256),
    ConvConfig(batch_size, 7, 48, 832, 7, 48),
    # ConvConfig(batch_size, 13, 256, 64, 13, 256),
    ConvConfig(batch_size, 14, 112, 512, 14, 112),
    ConvConfig(batch_size, 14, 160, 512, 14, 160),
    ConvConfig(batch_size, 14, 192, 480, 14, 192),
    ConvConfig(batch_size, 27, 256, 528, 27, 256),
    ConvConfig(batch_size, 14, 96, 480, 14, 96),
    ConvConfig(batch_size, 14, 256, 528, 14, 256),
    ConvConfig(batch_size, 14, 96, 480, 14, 96),
    # ConvConfig(batch_size, 27, 256, 64, 27, 256),
    ConvConfig(batch_size, 27, 48, 384, 27, 48),
    ConvConfig(batch_size, 28, 16, 192, 28, 16),
    ConvConfig(batch_size, 28, 64, 192, 28, 64),
    ConvConfig(batch_size, 28, 64, 192, 28, 64),
    # ConvConfig(batch_size, 55, 16, 128, 55, 16),
    # ConvConfig(batch_size, 55, 64, 16, 55, 64),
    ConvConfig(batch_size, 7, 160, 832, 7, 160),
    # ConvConfig(batch_size, 7, 256, 832, 7, 256),
]


def measure_time(model, input_data, kernel_data, times=TIMES):
    model = model.cuda().eval()
    input_data = input_data.cuda()
    kernel_data = kernel_data.cuda()

    with torch.no_grad():
        result = model(input_data)  # warmup and get result
        torch.cuda.synchronize()

        if times == 0:
            return 0, result

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()  # type: ignore
        for _ in range(times):
            model(input_data)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        end.record()  # type: ignore
        torch.cuda.synchronize()
    return start.elapsed_time(end) / times, result


def find_trec_param(trec, input_channels, output_channels, kernel_size, input_data, kernel_data):
    factors = divisors(input_channels * kernel_size * kernel_size)
    factors = [f for f in factors if f <= 8]
    return factors[-1], 8

    # best_param = (float('inf'), (None, None))

    # for param_L in range(1, 9):
    #     for param_H in range(1, 9):
    #         if input_channels * kernel_size * kernel_size % param_L == 0:
    #             try:
    #                 t, _ = measure_time(trec(input_channels, output_channels, kernel_size, param_L, param_H, layer=0),
    #                                     input_data, kernel_data, times=3)
    #             except Exception as e:
    #                 print(e)
    #                 continue

    #             print(f'i: {input_channels}, o: {output_channels}, k: '
    #                   f'{kernel_size}, L: {param_L}, H: {param_H}, time: {t}')
    #             if t < best_param[0]:
    #                 best_param = (t, (param_L, param_H))
    # return best_param[1]


def cutlass_convolution(input_data, kernel_data, config: ConvConfig, times=TIMES):
    N, C, H, W = input_data.shape
    K, R, S = config.output_channels, config.kernel_size, config.kernel_size
    stride = (1, 1)
    padding = (0, 0)
    dilation = (1, 1)

    P, Q = cutlass.Conv2d.output_size(
        (N, H, W, C), (K, R, S, C), padding, stride, dilation)[1:3]

    dtype = torch.float32
    input_data = input_data.to(memory_format=torch.channels_last).cuda()
    kernel_data = kernel_data.to(memory_format=torch.channels_last).cuda()
    tensor_C = torch.empty((N, K, P, Q), dtype=dtype,
                           device="cuda").to(memory_format=torch.channels_last)
    output = torch.zeros_like(tensor_C)

    alpha = 1.0
    beta = 0.0

    plan = cutlass.Conv2dFprop(element=dtype)

    plan.run(input_data, kernel_data, tensor_C, output, stride,
             padding, dilation, alpha, beta, print_module=False)
    torch.cuda.synchronize()

    if times == 0:
        return 0, output

    # Measure CUTLASS convolution time
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()  # type: ignore

    for _ in range(times):
        plan.run(input_data, kernel_data, tensor_C, output, stride,
                 padding, dilation, alpha, beta, print_module=False)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    end.record()  # type: ignore
    torch.cuda.synchronize()
    cutlass_time = start.elapsed_time(end) / times

    print(f'CUTLASS time: {cutlass_time}')

    return cutlass_time, output


results = {
    # 'cudnn': [],
    # 'cudnn_deterministic': [],
    'cudnn_benchmark': [],
    'cutlass': [],
    'trec': [],
    'trec_classic': [],
    'torch': [],
}

colors = {
    # 'cudnn': 'blue',
    # 'cudnn_deterministic': 'orange',
    'cudnn_benchmark': 'green',
    'cutlass': 'red',
    'torch': 'purple',
    'trec': 'brown',
    'trec_classic': 'pink',
}

for config in configs:

    with torch.no_grad():
        print(f'Running config: {config}')

        input_data, kernel_data = generate_data(config)

        # assert torch.cuda.is_available() and torch.backends.cudnn.is_available()
        # conv = nn.Conv2d(config.input_channels, config.output_channels,
        #                  config.kernel_size, bias=False)
        # conv.weight.data = kernel_data.cuda()
        # conv_time, conv_result = measure_time(conv, input_data, kernel_data)
        # results['cudnn'].append(conv_time)

        runned_trec = False
        param_L, param_H = find_trec_param(
            Conv2d_TREC,
            config.input_channels, config.output_channels, config.kernel_size,
            input_data, kernel_data)
        if param_L is not None and param_H is not None:
            print(f'Found TREC parameters: {param_L}, {param_H}')
            with torch.no_grad():
                trec = Conv2d_TREC(config.input_channels, config.output_channels,
                                   config.kernel_size, param_L, param_H, bias=False, layer=0)

            trec_time, trec_result = measure_time(
                trec, input_data, kernel_data)
            results['trec'].append(trec_time)
            runned_trec = True
        else:
            print('Conv2d_TREC failed')
            results['trec'].append(0)

        runned_trec_classic = False
        trec_classic_result: torch.Tensor | None = None
        # param_L, param_H = find_trec_param(
        #     Conv2d_TREC_Classic,
        #     config.input_channels, config.output_channels, config.kernel_size,
        #     input_data, kernel_data)
        # if param_L is not None and param_H is not None:
        #     print(f'Found TREC Classic parameters: {param_L}, {param_H}')
        #     with torch.no_grad():
        #         trec_classic = Conv2d_TREC_Classic(config.input_channels, config.output_channels,
        #                                            config.kernel_size, param_L, param_H, bias=False, layer=1)
        #         trec_classic.random_vectors = trec.random_vectors
        #         trec_classic.weight = trec.weight

        #     trec_classic_time, trec_classic_result = measure_time(
        #         trec_classic, input_data, kernel_data)
        #     results['trec_classic'].append(trec_classic_time)
        #     runned_trec_classic = True
        # else:
        #     print('Conv2d_TREC_Classic failed')
        results['trec_classic'].append(0)

        try:
            cutlass_time, cutlass_result = cutlass_convolution(
                input_data, kernel_data, config)
            results['cutlass'].append(cutlass_time)
        except Exception as e:
            print('CUTLASS failed, error:', e)
            results['cutlass'].append(0)

        torch.backends.cudnn.enabled = False
        conv = nn.Conv2d(config.input_channels, config.output_channels,
                         config.kernel_size, bias=False)
        conv_slow_time, conv_result = measure_time(
            conv, input_data, kernel_data)
        results['torch'].append(conv_slow_time)
        torch.backends.cudnn.enabled = True

        torch.backends.cudnn.benchmark = True
        conv = nn.Conv2d(config.input_channels, config.output_channels,
                         config.kernel_size, bias=False)
        conv_benchmark_time, _ = measure_time(conv, input_data, kernel_data)
        results['cudnn_benchmark'].append(conv_benchmark_time)
        torch.backends.cudnn.benchmark = False

        # torch.use_deterministic_algorithms(True)
        # conv = nn.Conv2d(config.input_channels, config.output_channels,
        #                  config.kernel_size, bias=False)
        # conv_deterministic_time, _ = measure_time(
        #     conv, input_data, kernel_data)
        # results['cudnn_deterministic'].append(conv_deterministic_time)
        # torch.use_deterministic_algorithms(False)

        # if runned_trec:
        #     if trec_result.shape == conv_result.shape:
        #         print(f'Conv2d_TREC error: {torch.abs(
        #             trec_result - conv_result).mean().item()}')
        #     else:
        #         print('Conv2d_TREC and Conv2d have different shapes')

        if runned_trec and runned_trec_classic:
            assert trec_classic_result is not None
            if trec_classic_result.shape == trec_result.shape:
                if not torch.allclose(trec_classic_result, trec_result, atol=1e-4):
                    print(f'Conv2d_TREC_Classic error: '
                          f'{torch.abs(trec_classic_result - trec_result).mean().item()}')  # type: ignore
            else:
                print(f"Conv2d_TREC_Classic and Conv2d_TREC have different shapes: "
                      f"{trec_classic_result.shape} vs {trec_result.shape}")  # type: ignore

        print()

x = np.arange(len(configs))
width = 0.1
fig, ax = plt.subplots(figsize=(15, 8))

for i, (method, times) in enumerate(results.items()):
    ax.bar(x + i * width, times, width, label=method, color=colors[method])

ax.set_ylabel('Time (ms)')
ax.set_title('Convolution Methods Performance Comparison')
ax.set_xticks(x + width * 2.5)
ax.set_xticklabels([f'({config.batch_size}x{config.input_channels}x{config.height}x{config.width}x'
                    f'{config.output_channels}x{config.kernel_size})' for config in configs], rotation=90)
ax.legend()

plt.tight_layout()
plt.show()
if TIMES != 0:
    plt.savefig(f'conv_perf_{"train" if TRAINING else "test"}.png')
