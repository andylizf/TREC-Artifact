
import torch.nn as nn
from conv_layer import Conv2d_TREC


class ZFNet(nn.Module):
    def __init__(self, params_L, params_H, dr=[0]*5, num_classes=10):
        super(ZFNet, self).__init__()

        # layer 1
        if dr[0]:
            self.conv1 = Conv2d_TREC(
                3, 96, 7, params_L[0], params_H[0], 0, stride=2, padding=1)
        else:
            self.conv1 = nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=1)
        self.relu1 = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, return_indices=True)

        # layer 2
        if dr[1]:
            self.conv2 = Conv2d_TREC(
                96, 256, 5, params_L[1], params_H[1], 1, stride=2)
        else:
            self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, return_indices=True)

        # layer 3
        if dr[2]:
            self.conv3 = Conv2d_TREC(
                256, 384, 3, params_L[2], params_H[2], 2, stride=1, padding=1)
        else:
            self.conv3 = nn.Conv2d(
                256, 384, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU(inplace=True)

        # layer 4
        if dr[3]:
            self.conv4 = Conv2d_TREC(
                384, 384, 3, params_L[3], params_H[3], 3, stride=1, padding=1)
        else:
            self.conv4 = nn.Conv2d(
                384, 384, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU(inplace=True)

        # layer 5
        if dr[4]:
            self.conv5 = Conv2d_TREC(
                384, 256, 3, params_L[4], params_H[4], 4, stride=1, padding=1)
        else:
            self.conv5 = nn.Conv2d(
                384, 256, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2, return_indices=True)

        self.classifier = nn.Sequential(
            # layer 6
            nn.Linear(9216, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            # layer 7
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x, indices = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x, indices = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x, indices = self.pool5(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
