import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
import os
from parse_timing import parse_timing_log

def test_L_value(L, batch_size=256, in_channels=48, out_channels=48, input_size=32):
    """运行单个L值的测试"""
    cmd = [
        'python', '-u', 'run_a_layer.py',
        '--param-L', str(L),
        '--param-H', str(12),
        '--batch-size', str(batch_size),
        '--in-channels', str(in_channels),
        '--out-channels', str(out_channels),
        '--input-size', str(input_size),
        '--num-warmup', '10',
        '--num-runs', '100'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, env=os.environ)
    return result.stdout

def test_baseline(batch_size=256, in_channels=48, out_channels=48, input_size=64):
    """运行普通卷积层作为基准"""
    cmd = [
        'python', '-u', 'run_a_layer.py',
        '--batch-size', str(batch_size),
        '--in-channels', str(in_channels),
        '--out-channels', str(out_channels),
        '--input-size', str(input_size),
        '--num-warmup', '10',
        '--num-runs', '100',
        '--baseline'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, env=os.environ)
    return result.stdout

def analyze_L_performance():
    # 生成合理范围的L值
    in_channels = 48
    kernel_size = 19
    max_L = min(26, in_channels * kernel_size * kernel_size)
    L_values = list(range(1, max_L + 1))
    
    # 存储结果
    all_results = []
    
    print("Testing different L values...")
    for L in L_values:
        print(f"\nTesting L={L}")
        output = test_L_value(L)
        
        # 使用parse_timing_log解析时间数据，但不打印中间统计
        time_stats = parse_timing_log(StringIO(output), print_stats=False)
        
        # 计算矩阵数量
        n_matrices = (in_channels * kernel_size * kernel_size + L - 1) // L
        
        # 存储结果
        result = {'L': L, 'n_matrices': n_matrices}
        
        # 添加每个阶段的平均时间
        for name, times in time_stats.items():
            if times:
                times = np.array(times)
                result[name] = np.mean(times)  # 保持原始单位（秒）
        
        # 计算总时间
        total_time = sum(result[name] for name in time_stats.keys() if name in result)
        result['Total Time'] = total_time
        
        all_results.append(result)
        
        # 打印当前L值的结果
        print(f"L={L}: Total Time={total_time*1000:.3f}ms, Matrices={n_matrices}")
    
    # 运行基准测试
    print("\nRunning baseline test...")
    baseline_output = test_baseline()
    baseline_stats = parse_timing_log(StringIO(baseline_output), print_stats=False)
    baseline_time = sum(np.mean(times) for times in baseline_stats.values()) * 1000  # 转换为毫秒
    
    # 创建DataFrame
    df = pd.DataFrame(all_results)
    
    # 转换时间单位为毫秒
    time_columns = [col for col in df.columns if col not in ['L', 'n_matrices']]
    df[time_columns] = df[time_columns] * 1000  # 转换为毫秒
    
    # 保存结果
    df.to_csv('L_performance_results.csv', index=False)
    
    # 绘制性能图表
    plt.figure(figsize=(15, 10))
    
    # 总时间 vs L (添加基准线)
    plt.subplot(2, 2, 1)
    plt.plot(df['L'], df['Total Time'], marker='o', label='TREC')
    plt.axhline(y=baseline_time, color='r', linestyle='--', label='Baseline Conv2d')
    plt.title('Total Time vs L')
    plt.xlabel('L value')
    plt.ylabel('Time (ms)')
    plt.legend()
    plt.grid(True)
    
    # 矩阵数量 vs L
    plt.subplot(2, 2, 2)
    plt.plot(df['L'], df['n_matrices'], marker='o')
    plt.title('Number of Matrices vs L')
    plt.xlabel('L value')
    plt.ylabel('Number of Matrices')
    plt.grid(True)
    
    # 主要计算阶段时间 vs L
    plt.subplot(2, 2, 3)
    key_stages = [
        'im2row_DRbatch_cuda',
        'input_row transpose',
        'matmul random vectors',
        'get_centroids_add_cuda',
        'get_id_count_cuda',
        'matrix multiplication'
    ]
    for stage in key_stages:
        if stage in df.columns:
            plt.plot(df['L'], df[stage], marker='o', label=stage)
    plt.title('Key Stage Times vs L')
    plt.xlabel('L value')
    plt.ylabel('Time (ms)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    
    # 所有阶段时间堆叠图
    plt.subplot(2, 2, 4)
    stages = [col for col in df.columns if col not in ['L', 'n_matrices', 'Total Time', 'Total forward pass']]
    if stages:  # 只在有阶段数据时绘制
        plt.stackplot(df['L'], [df[stage] for stage in stages], labels=stages)
        plt.title('Stage Times Distribution')
        plt.xlabel('L value')
        plt.ylabel('Time (ms)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.text(0.5, 0.5, 'No stage timing data available', 
                ha='center', va='center')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('L_performance_analysis.png', bbox_inches='tight')
    
    # 打印关键发现
    print("\nKey Findings:")
    best_L = df.loc[df['Total Time'].idxmin(), 'L']
    best_time = df.loc[df['L'] == best_L, 'Total Time'].iloc[0]
    print(f"Best performing L value: {best_L}")
    print(f"Number of matrices at best L: {df.loc[df['L'] == best_L, 'n_matrices'].iloc[0]}")
    print(f"Best total time: {best_time:.2f} ms")
    print(f"Baseline Conv2d time: {baseline_time:.2f} ms")
    print(f"Speedup vs baseline: {baseline_time/best_time:.2f}x")
    
    # 打印详细的性能数据表
    print("\nDetailed Performance Data:")
    print(df.to_string(index=False))

if __name__ == "__main__":
    analyze_L_performance()
