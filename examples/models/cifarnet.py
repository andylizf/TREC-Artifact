"""Contains a variant of the CIFAR-10 model definition."""

import torch.nn as nn
from conv_layer import Conv2d_TREC


class CifarNet_TREC(nn.Module):
    def __init__(self, params_L, params_H, dr=[0]*2, cpu=False):
        super(CifarNet_TREC, self).__init__()
        if dr[0]:
            self.conv1 = Conv2d_TREC(3, 64, 5, params_L[0], params_H[0], 0)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=5)
        self.batch1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        if dr[1]:
            self.conv2 = Conv2d_TREC(64, 64, 5, params_L[1], params_H[1], 1)
        else:
            self.conv2 = nn.Conv2d(64, 64, kernel_size=5)
        self.batch2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc3 = nn.Linear(1600, 384)
        self.dropout3 = nn.Dropout()
        self.relu3 = nn.ReLU(True)
        self.fc4 = nn.Linear(384, 192)
        self.dropout4 = nn.Dropout()
        self.relu4 = nn.ReLU(True)
        self.fc5 = nn.Linear(192, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.batch1(self.conv1(x))))
        x = self.pool2(self.relu2(self.batch2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.dropout3(self.fc3(x)))
        x = self.relu4(self.dropout4(self.fc4(x)))
        x = self.fc5(x)
        return x
