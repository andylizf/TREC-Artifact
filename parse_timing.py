import re
import numpy as np
import sys

def parse_timing_log(file):
    # Dictionary to store timing data
    time_stats = {
        'im2row_DRbatch_cuda': [],
        'input_row transpose': [],
        'matmul random vectors': [],
        'get_id_count_cuda': [],
        'index_bucket_cuda': [],
        'get_bucket_compact_ids_cuda': [],
        'get_centroids_add_cuda': [],
        'div_remap_centroids_cuda': [],
        'matrix multiplication': [],
        'reconstruct_output_cuda': [],
        'bias_add_cuda': [],
        'get_bucket_counts_out_cuda': []
    }
    
    # Read from file or stdin
    for line in file:
        if ':' in line and 'layer' not in line:
            try:
                name, time_str = [x.strip() for x in line.split(':')]
                if name in time_stats:
                    time_stats[name].append(float(time_str))
            except:
                continue
    
    # Print statistics
    print("Performance Statistics:")
    print("-" * 50)
    
    total_time = 0
    for name, times in time_stats.items():
        if times:
            times = np.array(times)
            mean_time = np.mean(times) * 1000  # Convert to ms
            std_time = np.std(times) * 1000
            total_time += mean_time
            print(f"{name:25s}: {mean_time:8.3f} ± {std_time:6.3f} ms")
    
    print("-" * 50)
    print(f"Total component time: {total_time:.3f} ms")
    
    # Print sample counts
    print("\nSamples collected for each stage:")
    for name, times in time_stats.items():
        print(f"{name:25s}: {len(times):4d} samples")

if __name__ == "__main__":
    # If a file is specified, read from it, otherwise read from stdin
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as f:
            parse_timing_log(f)
    else:
        parse_timing_log(sys.stdin) 