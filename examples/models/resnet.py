'''
resnet for cifar in pytorch

Reference:
[1] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.
[2] K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. In ECCV, 2016.
'''

import math

import torch
import torch.nn as nn
from trec.conv_layer import Conv2d_TREC

def conv1x1(in_planes, out_planes, stride=1, param_L=None, param_H=None, layer=None):
    """1x1 convolution, optionally with TREC"""
    if param_L is None:
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    return Conv2d_TREC(in_planes, out_planes, kernel_size=1, stride=stride, bias=False,
                      param_L=param_L, param_H=param_H, layer=layer)

def conv3x3(in_planes, out_planes, stride=1, param_L=None, param_H=None, layer=None):
    """3x3 convolution with padding, optionally with TREC"""
    if param_L is None:
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
    return Conv2d_TREC(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False,
                      param_L=param_L, param_H=param_H, layer=layer)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, 
                 param_L=None, param_H=None, layer=None):
        super(BasicBlock, self).__init__()
        
        # 分离参数给两个卷积层
        param_L0 = param_L[0] if param_L is not None else None
        param_H0 = param_H[0] if param_H is not None else None
        param_L1 = param_L[1] if param_L is not None else None
        param_H1 = param_H[1] if param_H is not None else None
        layer0 = layer[0] if layer is not None else None
        layer1 = layer[1] if layer is not None else None

        self.conv1 = conv3x3(inplanes, planes, stride, param_L0, param_H0, layer0)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1, param_L1, param_H1, layer1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class PreActBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(PreActBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(out)

        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += residual

        return out


class PreActBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(out)

        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out += residual

        return out


class ResNet_Cifar(nn.Module):
    def __init__(self, block, layers, num_classes=10, 
                 param_L=None, param_H=None, trec=None):
        super(ResNet_Cifar, self).__init__()
        self.inplanes = 16
        
        # 初始层的TREC参数
        init_param_L = param_L[0] if param_L is not None else None
        init_param_H = param_H[0] if param_H is not None else None
        
        # 根据是否使用TREC选择初始卷积
        if trec and trec[0]:
            self.conv1 = Conv2d_TREC(3, 16, kernel_size=3, stride=1, padding=1, bias=False,
                                   param_L=init_param_L, param_H=init_param_H, layer=0)
        else:
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
            
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        # 计算每层的参数索引
        param_idx = 1  # 从1开始，因为0已经被初始卷积使用
        
        # 构建三个stage
        self.layer1 = self._make_layer(block, 16, layers[0], 1,
                                     param_L[param_idx:param_idx+2*layers[0]] if param_L else None,
                                     param_H[param_idx:param_idx+2*layers[0]] if param_H else None,
                                     trec[param_idx:param_idx+2*layers[0]] if trec else None,
                                     param_idx)
        param_idx += 2*layers[0]
        
        self.layer2 = self._make_layer(block, 32, layers[1], 2,
                                     param_L[param_idx:param_idx+2*layers[1]] if param_L else None,
                                     param_H[param_idx:param_idx+2*layers[1]] if param_H else None,
                                     trec[param_idx:param_idx+2*layers[1]] if trec else None,
                                     param_idx)
        param_idx += 2*layers[1]
        
        self.layer3 = self._make_layer(block, 64, layers[2], 2,
                                     param_L[param_idx:param_idx+2*layers[2]] if param_L else None,
                                     param_H[param_idx:param_idx+2*layers[2]] if param_H else None,
                                     trec[param_idx:param_idx+2*layers[2]] if trec else None,
                                     param_idx)

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, Conv2d_TREC)):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, planes, blocks, stride=1, param_L=None, param_H=None, trec=None, param_idx=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # Downsample也需要支持TREC
            if trec and trec[0]:
                assert param_L is not None and param_H is not None, "param_L and param_H must be provided"
                downsample = nn.Sequential(
                    Conv2d_TREC(self.inplanes, planes * block.expansion, kernel_size=1, 
                              stride=stride, bias=False,
                              param_L=param_L[0], param_H=param_H[0], layer=param_idx),
                    nn.BatchNorm2d(planes * block.expansion)
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, 
                             stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion)
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                          param_L[:2] if param_L else None,
                          param_H[:2] if param_H else None,
                          [param_idx, param_idx+1] if trec else None))
        
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                              param_L=param_L[i*2:(i+1)*2] if param_L else None,
                              param_H=param_H[i*2:(i+1)*2] if param_H else None,
                              layer=[param_idx+i*2, param_idx+i*2+1] if trec else None))

        return nn.Sequential(*layers)

class PreAct_ResNet_Cifar(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(PreAct_ResNet_Cifar, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.bn = nn.BatchNorm2d(64*block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes*block.expansion,
                          kernel_size=1, stride=stride, bias=False)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes*block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet20_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [3, 3, 3], **kwargs)
    return model

def resnet32_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [5, 5, 5], **kwargs)
    return model


def resnet44_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [7, 7, 7], **kwargs)
    return model


def resnet56_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [9, 9, 9], **kwargs)
    return model


def resnet110_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [18, 18, 18], **kwargs)
    return model


def resnet1202_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [200, 200, 200], **kwargs)
    return model


def resnet164_cifar(**kwargs):
    model = ResNet_Cifar(Bottleneck, [18, 18, 18], **kwargs)
    return model


def resnet1001_cifar(**kwargs):
    model = ResNet_Cifar(Bottleneck, [111, 111, 111], **kwargs)
    return model


def preact_resnet110_cifar(**kwargs):
    model = PreAct_ResNet_Cifar(PreActBasicBlock, [18, 18, 18], **kwargs)
    return model


def preact_resnet164_cifar(**kwargs):
    model = PreAct_ResNet_Cifar(PreActBottleneck, [18, 18, 18], **kwargs)
    return model


def preact_resnet1001_cifar(**kwargs):
    model = PreAct_ResNet_Cifar(PreActBottleneck, [111, 111, 111], **kwargs)
    return model
