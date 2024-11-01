import torch
import sys
import os

def test_row2im(layer_num):
    # 加载保存的数据
    data = torch.load(f'./examples/debug_row2im_layer_{layer_num}.pt', weights_only=True)
    
    for name, tensor in data.items():
        if isinstance(tensor, torch.Tensor):
            print(f"{name}: shape={tensor.shape}, "
                    f"device={tensor.device}, "
                    f"range=[{tensor.min():.6f}, {tensor.max():.6f}]")
        else:
            print(f"{name}: {tensor}")
    from trec._C import conv_deep_reuse_backward
    try:
        grads = conv_deep_reuse_backward(
            data['input_row'],
            data['inputCentroids'],
            data['weights'],
            data['gradOutput'],
            data['vector_index'],
            data['vector_ids'],
            data['buckets_count'],
            data['buckets_index'],
            data['buckets_index_inv'],
            data['random_vectors'],
            data['input_height'],
            data['input_width'],
            data['padding'][0],
            data['padding'][1],
            data['stride'][0],
            data['stride'][1],
            data['param_H'],
            data['alpha'],
            data['sigma'],
            data['do_bias']
        )
        print("Success!")
        return grads
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_row2im.py <layer_number>")
        sys.exit(1)
    
    layer_num = int(sys.argv[1])
    result = test_row2im(layer_num)