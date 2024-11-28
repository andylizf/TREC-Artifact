import math

def calculate_valid_param_L(in_channels, kernel_size, min_L=1, max_L=64):
    """
    计算有效的 param_L 值
    
    Args:
        in_channels: 输入通道数
        kernel_size: 卷积核大小
        min_L: 最小的 param_L 值
        max_L: 最大的 param_L 值
    
    Returns:
        list: 所有有效的 param_L 值及其相关信息
    """
    # 计算 row_length
    row_length = in_channels * kernel_size * kernel_size
    
    print(f"Input parameters:")
    print(f"in_channels: {in_channels}")
    print(f"kernel_size: {kernel_size}")
    print(f"row_length: {row_length}")
    print("\nAnalyzing possible param_L values:")
    print("-" * 80)
    print(f"{'param_L':>8} | {'n_matrices':>10} | {'shared_mem':>10} | {'status':>30}")
    print("-" * 80)
    
    valid_values = []
    
    # 遍历可能的 param_L 值
    for L in range(min_L, min(max_L + 1, row_length + 1)):
        # 计算需要的矩阵数量（向上取整）
        n_matrices = (row_length + L - 1) // L
        
        # 计算共享内存使用量 (假设 float32)
        vector_dim = L
        max_buckets = 256  # 2^8 (param_H = 8)
        fixed_mem = max_buckets * vector_dim * 4  # 4 bytes per float
        
        status = "OK"
        if fixed_mem > 48*1024:  # 48KB shared memory limit
            status = "Exceeds shared memory"
        else:
            valid_values.append({
                'param_L': L,
                'n_matrices': n_matrices,
                'shared_mem': fixed_mem
            })
        
        print(f"{L:8d} | {n_matrices:10d} | {fixed_mem:10d} | {status:>30}")
    
    print("\nRecommended param_L values:")
    print("-" * 80)
    if valid_values:
        # 按性能指标排序（矩阵数量和共享内存使用）
        sorted_values = sorted(valid_values, 
                             key=lambda x: (x['n_matrices'], x['shared_mem']))
        
        print("Top 10 recommendations (sorted by number of matrices and shared memory usage):")
        for v in sorted_values[:10]:
            print(f"param_L = {v['param_L']:3d}: n_matrices = {v['n_matrices']:4d}, "
                  f"shared memory = {v['shared_mem']/1024:.1f}KB")
    else:
        print("No valid param_L values found!")
    
    return valid_values

if __name__ == "__main__":
    # 使用默认配置
    in_channels = 48
    kernel_size = 3
    
    valid_values = calculate_valid_param_L(in_channels, kernel_size)
    
    if valid_values:
        print("\nSuggested configurations:")
        # 选择几个不同的推荐值
        best_matrices = min(valid_values, key=lambda x: x['n_matrices'])
        best_memory = min(valid_values, key=lambda x: x['shared_mem'])
        balanced = min(valid_values, 
                      key=lambda x: x['n_matrices'] * x['shared_mem'])
        
        print(f"""
Different optimization targets:

1. Minimum matrices (fastest):
param_L = {best_matrices['param_L']}
n_matrices = {best_matrices['n_matrices']}
shared_memory = {best_matrices['shared_mem']/1024:.1f}KB

2. Minimum shared memory:
param_L = {best_memory['param_L']}
n_matrices = {best_memory['n_matrices']}
shared_memory = {best_memory['shared_mem']/1024:.1f}KB

3. Balanced performance:
param_L = {balanced['param_L']}
n_matrices = {balanced['n_matrices']}
shared_memory = {balanced['shared_mem']/1024:.1f}KB

Example configuration:
conv_trec = Conv2d_TREC(
    in_channels={in_channels},
    out_channels=12,
    kernel_size={kernel_size},
    param_L={balanced['param_L']},  # Balanced value
    param_H=8,
    layer=0,
    padding=1,
    stride=1
)
""") 