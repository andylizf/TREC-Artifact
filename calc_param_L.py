import math

def calculate_valid_param_L(in_channels, kernel_size, min_L=3, max_L=512):
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
    print("-" * 70)
    print(f"{'param_L':>8} | {'n_matrices':>10} | {'aligned':>7} | {'shared_mem':>10} | {'status':>20}")
    print("-" * 70)
    
    valid_values = []
    
    # 遍历可能的 param_L 值
    for L in range(min_L, min(max_L + 1, row_length + 1)):
        # 检查是否是 row_length 的因子
        if row_length % L != 0:
            continue
        
        n_matrices = row_length // L
        
        # 检查是否与 warp size (32) 对齐
        is_aligned = n_matrices % 32 == 0
        
        # 计算共享内存使用量 (假设 float32)
        vector_dim = L
        max_buckets = 256  # 2^8 (param_H = 8)
        fixed_mem = max_buckets * vector_dim * 4  # 4 bytes per float
        
        status = "OK"
        if fixed_mem > 16384:  # 16KB shared memory limit
            status = "Exceeds shared memory"
        elif not is_aligned:
            status = "Not warp aligned"
        else:
            valid_values.append({
                'param_L': L,
                'n_matrices': n_matrices,
                'shared_mem': fixed_mem,
                'aligned': is_aligned
            })
        
        print(f"{L:8d} | {n_matrices:10d} | {str(is_aligned):>7} | {fixed_mem:10d} | {status:>20}")
    
    print("\nRecommended param_L values:")
    print("-" * 70)
    if valid_values:
        for v in valid_values:
            print(f"param_L = {v['param_L']:3d}: n_matrices = {v['n_matrices']:4d}, shared memory = {v['shared_mem']/1024:.1f}KB")
    else:
        print("No valid param_L values found!")
    
    return valid_values

if __name__ == "__main__":
    # 使用你的配置
    in_channels = 48
    kernel_size = 3
    
    valid_values = calculate_valid_param_L(in_channels, kernel_size)
    
    if valid_values:
        print("\nSuggested configuration:")
        # 选择最佳值（这里选择最大的符合条件的 param_L）
        best = max(valid_values, key=lambda x: x['param_L'])
        print(f"""
Best param_L = {best['param_L']}

Example configuration:
conv_trec = Conv2d_TREC(
    in_channels={in_channels},
    out_channels=12,
    kernel_size={kernel_size},
    param_L={best['param_L']},  # Optimized value
    param_H=8,
    layer=0,
    padding=1,
    stride=1
)
""") 