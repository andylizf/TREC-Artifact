import re
import numpy as np
import sys
from collections import defaultdict

def parse_timing_log(file, print_stats=True):
    """Parse timing information from log file
    Args:
        file: File object containing timing logs
        print_stats: Whether to print statistics (default: True)
    Returns:
        Dictionary containing timing statistics
    """
    time_stats = defaultdict(list)
    
    # Read from file or stdin
    for line in file:
        if ':' in line:
            try:
                # 使用maxsplit=1确保只在第一个冒号处分割
                name, time_str = line.split(':', maxsplit=1)
                name = name.strip()
                
                # Skip non-timing information and configuration lines
                if any(x in name.lower() for x in ['config', 'shape', 'memory', 'layer', 'average time']):
                    continue
                    
                # 尝试提取时间值
                try:
                    time_val = float(time_str.split()[0])
                    time_stats[name].append(time_val)
                except ValueError:
                    continue
                    
            except ValueError:
                continue
    
    if print_stats and time_stats:
        print("Performance Statistics:")
        print("-" * 50)
        
        total_time = 0
        for name, times in time_stats.items():
            if times:
                times = np.array(times)
                mean_time = np.mean(times) * 1000  # Convert to ms
                std_time = np.std(times) * 1000
                total_time += mean_time
                print(f"{name:30s}: {mean_time:8.3f} ± {std_time:6.3f} ms")
        
        print("-" * 50)
        print(f"Total component time: {total_time:.3f} ms")
        
        print("\nSamples collected for each metric:")
        for name, times in time_stats.items():
            print(f"{name:30s}: {len(times):4d} samples")
    
    return time_stats

if __name__ == "__main__":
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as f:
            parse_timing_log(f)
    else:
        parse_timing_log(sys.stdin) 