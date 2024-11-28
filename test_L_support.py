from run_a_layer import profile_conv_trec

# 使用不能被L整除的配置运行测试
conv_trec, inputs, output = profile_conv_trec(
    batch_size=2,
    in_channels=3,
    out_channels=16,
    input_size=32,
    kernel_size=3,
    param_L=7,  # 不能整除 in_channels * kernel_size * kernel_size (=27)
    param_H=8,
    num_warmup=1,
    num_runs=1,
    do_backward=True
)

# 打印额外的信息
print(f"\nAdditional Information:")
print(f"Row length (in_channels * kernel_size^2): {3 * 3 * 3}")
print(f"Number of matrices: {(3 * 3 * 3 + 9 - 1) // 9}")
print(f"Output contiguous: {output.is_contiguous()}")
