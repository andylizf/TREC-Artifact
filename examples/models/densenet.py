"""
Edited from https://github.com/ZhangXLaurence/pytorch-cifar-models/blob/master/models/densenet_cifar.py

DenseNet for cifar with pytorch

Reference:
[1] H. Gao, Z. Liu, L. Maaten and K. Weinberger. Densely connected convolutional networks. In CVPR, 2017
"""

import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from trec.conv_layer import Conv2d_TREC

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, 
                 params_L, params_H, trec, idx):
        super(_DenseLayer, self).__init__()
        
        if trec and trec[0]:
            self.add_module('conv1', Conv2d_TREC(num_input_features, bn_size * growth_rate, 
                                                 kernel_size=1, stride=1, bias=False,
                                                 param_L=params_L[0], param_H=params_H[0], layer=idx))
        else:
            self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate, 
                                               kernel_size=1, stride=1, bias=False))
        
        self.add_module('norm1', nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module('relu1', nn.ReLU(inplace=True))
        
        if trec and trec[1]:
            self.add_module('conv2', Conv2d_TREC(bn_size * growth_rate, growth_rate,
                                                 kernel_size=3, stride=1, padding=1, bias=False,
                                                 param_L=params_L[1], param_H=params_H[1], layer=idx+1))
        else:
            self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                               kernel_size=3, stride=1, padding=1, bias=False))
        
        self.add_module('norm2', nn.BatchNorm2d(growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        
        self.drop_rate = drop_rate
        self.idx = idx

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)

class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, 
                 params_L, params_H, trec, idx_offset):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate,
                                params_L[i*2:(i+1)*2] if params_L else None,
                                params_H[i*2:(i+1)*2] if params_H else None, 
                                trec[i*2:(i+1)*2] if trec else None,
                                idx_offset + i*2)
            self.add_module('denselayer%d' % (i + 1), layer)

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, params_L, params_H, trec, idx):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        if trec:
            self.add_module('conv', Conv2d_TREC(num_input_features, num_output_features,
                                                kernel_size=1, stride=1, bias=False,
                                                param_L=params_L, param_H=params_H, layer=idx))
        else:
            self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                              kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

class DenseNet_Cifar_TREC(nn.Module):
    def __init__(self, growth_rate=12, block_config=(16, 16, 16),
                 num_init_features=24, bn_size=4, drop_rate=0, num_classes=10,
                 params_L=None, params_H=None, trec=None):
        super(DenseNet_Cifar_TREC, self).__init__()

        num_trec_layers = 1 + sum(block_config) * 2 + len(block_config) - 1

        if trec and any(trec):
            assert params_L is not None and params_H is not None, "params_L and params_H must be provided when using TREC"
            assert len(params_L) == num_trec_layers, f"params_L must have {num_trec_layers} elements when using TREC"
            assert len(params_H) == num_trec_layers, f"params_H must have {num_trec_layers} elements when using TREC"
            assert len(trec) == num_trec_layers, f"trec must have {num_trec_layers} elements"

        if trec and trec[0]:
            assert params_L and params_H, "params_L and params_H must be provided if trec[0] is True"
            self.features = nn.Sequential(OrderedDict([
                ('conv0', Conv2d_TREC(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False,
                                      param_L=params_L[0], param_H=params_H[0], layer=0)),
            ]))
        else:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ]))

        self.features.add_module('norm0', nn.BatchNorm2d(num_init_features))
        self.features.add_module('relu0', nn.ReLU(inplace=True))

        num_features = num_init_features
        trec_idx = 1
        params_L_idx = 1
        params_H_idx = 1

        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate,
                                params_L=params_L[params_L_idx:params_L_idx+2*num_layers] if params_L else None,
                                params_H=params_H[params_H_idx:params_H_idx+2*num_layers] if params_H else None,
                                trec=trec[trec_idx:trec_idx+2*num_layers] if trec else None,
                                idx_offset=trec_idx)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            trec_idx += 2*num_layers
            params_L_idx += 2*num_layers
            params_H_idx += 2*num_layers

            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2,
                                    params_L=params_L[params_L_idx] if params_L else None,
                                    params_H=params_H[params_H_idx] if params_H else None,
                                    trec=trec[trec_idx] if trec else None, idx=trec_idx)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
                trec_idx += 1
                params_L_idx += 1
                params_H_idx += 1

        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        self.classifier = nn.Linear(num_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

def densenet_BC_cifar_TREC(depth, k, params_L, params_H, trec, **kwargs):
    N = (depth - 4) // 6
    model = DenseNet_Cifar_TREC(growth_rate=k, block_config=[N, N, N], num_init_features=2*k, 
                                params_L=params_L, params_H=params_H, trec=trec, **kwargs)
    return model

if __name__ == '__main__':
    # Example usage
    depth = 40
    k = 12
    num_classes = 10
    
    # Calculate the number of TREC layers
    N = (depth - 4) // 6
    num_trec_layers = 1 + N * 6 + 2  # Initial conv + 3 dense blocks with 2N layers each + 2 transition layers
    
    params_L = [
        9,   # 初始卷积 (3x3x3=27, 27%9=0)
        3, 9,   # 第一个密集块的第一层 (1x1, 3x3)
        3, 9,   # 第一个密集块的第二层
        3, 9,   # 第一个密集块的第三层
        3, 9,   # 第一个密集块的第四层
        3, 9,   # 第一个密集块的第五层
        3, 9,   # 第一个密集块的第六层
        6,   # 第一个过渡层
        6, 9,   # 第二个密集块的第一层
        6, 9,   # 第二个密集块的第二层
        6, 9,   # 第二个密集块的第三层
        6, 9,   # 第二个密集块的第四层
        6, 9,   # 第二个密集块的第五层
        6, 9,   # 第二个密集块的第六层
        6,   # 第二个过渡层
        6, 9,   # 第三个密集块的第一层
        6, 9,   # 第三个密集块的第二层
        6, 9,   # 第三个密集块的第三层
        6, 9,   # 第三个密集块的第四层
        6, 9,   # 第三个密集块的第五层
        6, 9,   # 第三个密集块的第六层
    ]
    params_H = [8] * num_trec_layers
    trec = [1] * num_trec_layers  # Set to 1 to use TREC for all layers, or customize as needed
    
    net = densenet_BC_cifar_TREC(depth, k, params_L=params_L, params_H=params_H, trec=trec, num_classes=num_classes)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = net.to(device)
    
    input_tensor = torch.randn(1, 3, 32, 32).to(device)
    
    output = net(input_tensor)
    
    print(net)
    print(f"Output shape: {output.shape}")