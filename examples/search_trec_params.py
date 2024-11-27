import os
import subprocess
import json
import time
from pathlib import Path

def generate_trec_config(position, total_layers=16):
    """生成TREC配置，在指定位置设置为1"""
    return ','.join(['1' if i == position else '0' for i in range(total_layers)])

def get_layer_params(position):
    """根据位置获取该层的参数"""
    base_ch = 64
    ch_mult = [1, 2, 4, 8]
    
    # 计算是encoder还是decoder
    num_encoder_layers = len(ch_mult) * 2
    is_encoder = position < num_encoder_layers
    
    # 计算block索引和block内的位置
    block_idx = (position % num_encoder_layers) // 2
    is_first_conv = (position % 2) == 0
    
    print(f"\nAnalyzing layer at position {position}:")
    print(f"Is encoder: {is_encoder}")
    print(f"Block index: {block_idx}")
    print(f"Is first conv in block: {is_first_conv}")
    
    if is_encoder:
        if block_idx == 0 and is_first_conv:
            in_channels = 3
            out_channels = base_ch * ch_mult[block_idx]
            print(f"First encoder block, first conv: 3 -> {out_channels}")
        else:
            in_channels = base_ch * ch_mult[block_idx - 1] if is_first_conv else base_ch * ch_mult[block_idx]
            out_channels = base_ch * ch_mult[block_idx]
            print(f"Encoder block {block_idx}, {'first' if is_first_conv else 'second'} conv: {in_channels} -> {out_channels}")
    else:
        decoder_block_idx = block_idx
        if decoder_block_idx == 0:
            in_channels = base_ch * ch_mult[-1]
            print(f"First decoder block: starting with {in_channels} channels")
        else:
            in_channels = base_ch * ch_mult[-decoder_block_idx]
            print(f"Decoder block {decoder_block_idx}")
        
        out_channels = base_ch * ch_mult[-decoder_block_idx-1] if decoder_block_idx < len(ch_mult)-1 else base_ch
        print(f"Decoder channels: {in_channels} -> {out_channels}")
    
    kernel_size = 3
    row_length = in_channels * kernel_size * kernel_size
    print(f"Computed row length: {row_length} (in_ch={in_channels} * k={kernel_size}^2)")
    
    return in_channels, out_channels, kernel_size

def get_valid_L_values(in_channels, kernel_size, min_L=3, max_L=512):
    """计算有效的param_L值"""
    row_length = in_channels * kernel_size * kernel_size
    valid_values = []
    
    print(f"\nCalculating valid L values:")
    print(f"Input channels: {in_channels}")
    print(f"Kernel size: {kernel_size}")
    print(f"Row length: {row_length}")
    print(f"Search range: {min_L} to {min(max_L + 1, row_length + 1)}")
    
    for L in range(min_L, min(max_L + 1, row_length + 1)):
        if row_length % L != 0:
            continue
        
        n_matrices = row_length // L
        is_aligned = n_matrices % 32 == 0
        shared_mem = 256 * L * 4  # max_buckets * vector_dim * sizeof(float)
        
        print(f"\nTesting L={L}:")
        print(f"  n_matrices: {n_matrices}")
        print(f"  warp aligned: {'✓' if is_aligned else '✗'}")
        print(f"  shared memory: {shared_mem/1024:.1f}KB")
        print(f"  shared memory limit: {'✓' if shared_mem <= 16384 else '✗'}")
        
        if shared_mem <= 16384:  # 只检查共享内存限制
            valid_values.append({
                'L': L,
                'is_aligned': is_aligned,
                'shared_mem': shared_mem,
                'n_matrices': n_matrices
            })
            print("  ✓ Valid value found!")
            if not is_aligned:
                print("  ⚠️ Warning: Not warp aligned, may impact performance")
    
    print(f"\nFound {len(valid_values)} valid L values:")
    for val in valid_values:
        print(f"  L={val['L']}: {'✓' if val['is_aligned'] else '✗'} warp aligned, "
              f"{val['shared_mem']/1024:.1f}KB shared mem, {val['n_matrices']} matrices")
    
    return valid_values

def run_experiment(position, param_L_info, exp_dir, gpu_id=0):
    """运行单个实验"""
    param_L = param_L_info['L']
    exp_name = f"trec_pos_{position}_L_{param_L}"
    if not param_L_info['is_aligned']:
        exp_name += "_unaligned"
    checkpoint_path = os.path.join(exp_dir, exp_name)
    
    # 生成配置
    trec_config = generate_trec_config(position)
    L_config = ','.join(['9' if i != position else str(param_L) for i in range(16)])
    H_config = ','.join(['8' for _ in range(16)])
    
    # 打印完整配置
    print("\nExperiment Configuration:", flush=True)
    print(f"Position: {position}", flush=True)
    print(f"param_L: {param_L}", flush=True)
    print(f"Warp aligned: {'✓' if param_L_info['is_aligned'] else '✗'}", flush=True)
    print("\nTREC configuration:", flush=True)
    print(f"TREC: {trec_config}", flush=True)
    print(f"L: {L_config}", flush=True)
    print(f"H: {H_config}", flush=True)
    
    # 构建命令
    cmd = [
        "python", "-u",  # 添加 -u 参数禁用 Python 缓冲
        "train_autoencoder.py",
        f"--model_name=autoencoder_trec",
        f"--checkpoint_path={checkpoint_path}",
        "--dataset_path=data",
        "--epochs=10",
        "--batch_size=32",
        "--learning_rate=1e-4",
        "--weight_decay=1e-6",
        f"--trec={trec_config}",
        f"--L={L_config}",
        f"--H={H_config}",
        f"--gpu={gpu_id}"
    ]
    
    print("\nCommand:", flush=True)
    print(" ".join(cmd), flush=True)
    
    # 创建日志目录
    log_dir = Path(exp_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{exp_name}.log"
    
    print(f"\nLogging to: {log_file}", flush=True)
    
    # 设置环境变量禁用缓冲
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    
    # 运行实验并记录日志
    with open(log_file, 'w', buffering=1) as f:
        start_time = time.time()
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,  # 行缓冲
            env=env
        )
        
        # 实时记录输出和解析loss
        best_loss = float('inf')
        for line in process.stdout:
            f.write(line)
            f.flush()
            print(line, end='', flush=True)  # 实时显示输出
            
            # 修改loss解析逻辑
            if "Loss:" in line and ", Loss:" in line:  # 确保是metrics输出行
                try:
                    loss = float(line.split(", Loss:")[-1].strip().split()[0])
                    best_loss = min(best_loss, loss)
                    print(f"Current loss: {loss}, Best loss so far: {best_loss}", flush=True)
                except Exception as e:
                    print(f"Error parsing loss: {e}", flush=True)
                    print(f"Line was: {line}", flush=True)
        
        process.wait()
        duration = time.time() - start_time
    
    return {
        'position': position,
        'param_L': param_L,
        'is_aligned': param_L_info['is_aligned'],
        'best_loss': best_loss,
        'duration': duration,
        'log_file': str(log_file)
    }

def main():
    exp_dir = "trec_search_results"
    gpu_id = 0
    results_file = os.path.join(exp_dir, 'results.json')
    
    print("\n" + "="*50, flush=True)
    print("Starting TREC parameter search", flush=True)
    print("="*50, flush=True)
    
    # 创建实验目录
    os.makedirs(exp_dir, exist_ok=True)
    
    # 加载已有结果（如果存在）
    results = []
    if os.path.exists(results_file):
        with open(results_file) as f:
            results = json.load(f)
            print(f"\nLoaded {len(results)} existing results")
    
    # 对每个位置进行实验
    total_layers = 16
    print(f"\nSearching through {total_layers} layers")
    
    for position in range(total_layers):
        print(f"\n{'-'*50}")
        print(f"Processing layer position {position}")
        print(f"{'-'*50}")
        
        # 获取该层的参数
        in_channels, out_channels, kernel_size = get_layer_params(position)
        
        # 获取该层可用的param_L值
        valid_L_values = get_valid_L_values(in_channels, kernel_size)
        
        if not valid_L_values:
            print(f"\nNo valid param_L values found for position {position}, skipping...")
            continue
        
        print(f"\nFound {len(valid_L_values)} valid configurations to test")
        
        for param_L_info in valid_L_values:
            if any(r['position'] == position and r['param_L'] == param_L_info['L'] for r in results):
                print(f"\nSkipping existing experiment: pos={position}, L={param_L_info['L']}")
                continue
            
            print(f"\nRunning experiment with position={position}, L={param_L_info['L']}")
            if not param_L_info['is_aligned']:
                print("⚠️ Warning: This configuration is not warp aligned")
            
            result = run_experiment(position, param_L_info, exp_dir, gpu_id)
            results.append(result)
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            best_result = min(results, key=lambda x: x['best_loss'])
            print("\nCurrent best configuration:")
            print(f"Position: {best_result['position']}")
            print(f"param_L: {best_result['param_L']}")
            print(f"Best loss: {best_result['best_loss']}")
    
    print("\n" + "="*50)
    print("Search completed!")
    print("="*50)
    print(f"Total experiments: {len(results)}")
    
    if results:
        print("\nTop 5 configurations:")
        top_5 = sorted(results, key=lambda x: x['best_loss'])[:5]
        for i, result in enumerate(top_5, 1):
            print(f"\n{i}. Loss: {result['best_loss']:.6f}")
            print(f"   Position: {result['position']}")
            print(f"   param_L: {result['param_L']}")
            print(f"   Warp aligned: {'✓' if result['is_aligned'] else '✗'}")
            print(f"   Duration: {result['duration']:.2f}s")
            print(f"   Log: {result['log_file']}")
    else:
        print("\nNo valid configurations found!")

if __name__ == "__main__":
    main() 