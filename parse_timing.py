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
        'get_bucket_counts_out_cuda': [],
        'Stream initialization': [],
        'Tensor allocation': [],
        'Batch start': [],
        'Batch processing': [],
        'Batch sync and merge': [],
        'Index bucket': [],
        'Output reshape': [],
        'Cleanup': [],
        'Training mode extra processing': [],
        'Input row reorganization': [],
        'Total forward pass': []
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
    print("\nPerformance Statistics:")
    print("-" * 60)
    print(f"{'Stage':35s}{'Mean':>12s}{'Std':>12s}")
    print("-" * 60)
    
    # 分类统计
    pipeline_time = 0
    processing_time = 0
    overhead_time = 0
    
    for name, times in time_stats.items():
        if times:
            times = np.array(times)
            mean_time = np.mean(times) * 1000  # Convert to ms
            std_time = np.std(times) * 1000
            
            # 分类累加时间
            if name in ['Stream initialization', 'Tensor allocation', 'Cleanup']:
                overhead_time += mean_time
            elif name in ['Batch processing', 'Batch sync and merge']:
                pipeline_time += mean_time
            elif name != 'Total forward pass':
                processing_time += mean_time
                
            print(f"{name:35s}{mean_time:10.3f} ±{std_time:8.3f} ms")
    
    print("-" * 60)
    print(f"{'Pipeline overhead':35s}{overhead_time:10.3f} ms")
    print(f"{'Pipeline processing':35s}{pipeline_time:10.3f} ms")
    print(f"{'Other processing':35s}{processing_time:10.3f} ms")
    print(f"{'Total time':35s}{(overhead_time + pipeline_time + processing_time):10.3f} ms")
    
    # Print sample counts
    print("\nSamples collected for each stage:")
    print("-" * 60)
    for name, times in time_stats.items():
        print(f"{name:35s}{len(times):6d} samples")

if __name__ == "__main__":
    # If a file is specified, read from it, otherwise read from stdin
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as f:
            parse_timing_log(f)
    else:
        parse_timing_log(sys.stdin) 