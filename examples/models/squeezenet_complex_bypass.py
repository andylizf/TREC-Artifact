
"""Contains the definition of the SqueezeNet + complex bypass architecture.

As described in https://arxiv.org/pdf/1602.07360.pdf?ref=https://githubhelp.com.

  SQUEEZENET: ALEXNET-LEVEL ACCURACY WITH 50X FEWER PARAMETERS AND <0.5MB MODEL SIZE
  Forrest N. Iandola, Song Han, Matthew W. Moskewicz, Khalid Ashraf, William J. Dally, Kurt Keutzer
"""

import sys
sys.path.append("../")
import torch
import torch.nn as nn
import torch.nn.functional as F
from timeit import default_timer as timer
from conv_layer import Conv2d_TREC


class Fire(nn.Module):

    def __init__(self, in_channel, out_channel, squzee_channel, 
                    params_L=[1]*3, params_H=[1]*3, trec=[0]*3, idx=0):

        super().__init__()
        if trec[0]:
            self.squeeze = nn.Sequential(
                Conv2d_TREC(in_channel, squzee_channel, 1, params_L[0], params_H[0], idx),
                nn.BatchNorm2d(squzee_channel),
                nn.ReLU(inplace=True)
            )
        else:
            self.squeeze = nn.Sequential(
                nn.Conv2d(in_channel, squzee_channel, 1),
                nn.BatchNorm2d(squzee_channel),
                nn.ReLU(inplace=True)
            )

        if trec[1]:
            self.expand_1x1 = nn.Sequential(
                Conv2d_TREC(squzee_channel, int(out_channel / 2), 1, params_L[1], params_H[1], idx+1),
                nn.BatchNorm2d(int(out_channel / 2)),
                nn.ReLU(inplace=True)
            )
        else:
            self.expand_1x1 = nn.Sequential(
                nn.Conv2d(squzee_channel, int(out_channel / 2), 1),
                nn.BatchNorm2d(int(out_channel / 2)),
                nn.ReLU(inplace=True)
            )

        if trec[2]:
            self.expand_3x3 = nn.Sequential(
                Conv2d_TREC(squzee_channel, int(out_channel / 2), 3, params_L[2], params_H[2], idx+2, padding=1),
                nn.BatchNorm2d(int(out_channel / 2)),
                nn.ReLU(inplace=True)
            )
        else:
            self.expand_3x3 = nn.Sequential(
                nn.Conv2d(squzee_channel, int(out_channel / 2), 3, padding=1),
                nn.BatchNorm2d(int(out_channel / 2)),
                nn.ReLU(inplace=True)
            )
    
    def forward(self, x):

        x = self.squeeze(x)
        x = torch.cat([
            self.expand_1x1(x),
            self.expand_3x3(x)
        ], 1)

        return x


class SqueezeNet_TREC(nn.Module):
    def __init__(self, params_L, params_H, bp_params_L, bp_params_H, trec=[0]*(8*3+2), bypass_trec=[0]*4, class_num=10):
        super().__init__()
        cfg = [[96, 128, 16], [128, 128, 16], [128, 256, 32], [256, 256, 32], [256, 384, 48], [384, 384, 48], [384, 512, 64], [512, 512, 64], [512, class_num, 1]]
        
        if trec[0]:
            self.stem = nn.Sequential(
            Conv2d_TREC(3, 96, 3, params_L[0], params_H[0], 0, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(3, 96, 3, padding=1),
                nn.BatchNorm2d(96),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            )
            
        self.Fires = nn.ModuleList()
        for i in range(8):
            if 1 in trec[i*3+1: i*3+4]:
                self.Fires.append(Fire(cfg[i][0], cfg[i][1], cfg[i][2], params_L[i*3+1 : i*3+4], params_H[i*3+1 : i*3+4], trec=trec[i*3+1: i*3+4], idx=i*3+1))
            else:
                self.Fires.append(Fire(cfg[i][0], cfg[i][1], cfg[i][2]))

        if bypass_trec[0]:
            self.bp_conv1 = Conv2d_TREC(96, 128, 1, bp_params_L[0], bp_params_H[0], 0)
        else:
            self.bp_conv1 = nn.Conv2d(96, 128, 1)

        if bypass_trec[1]:
            self.bp_conv2 = Conv2d_TREC(128, 256, 1, bp_params_L[1], bp_params_H[1], 1)
        else:
            self.bp_conv2 = nn.Conv2d(128, 256, 1)

        if bypass_trec[2]:
            self.bp_conv3 = Conv2d_TREC(256, 384, 1, bp_params_L[2], bp_params_H[2], 2)
        else:
            self.bp_conv3 = nn.Conv2d(256, 384, 1)

        if bypass_trec[3]:
            self.bp_conv4 = Conv2d_TREC(384, 512, 1, bp_params_L[3], bp_params_H[3], 3)
        else:
            self.bp_conv4 = nn.Conv2d(384, 512, 1)

        if trec[-1]:
            self.conv10 = Conv2d_TREC(512, class_num, 1, params_L[-1], params_H[-1], 25)
        else:
            self.conv10 = nn.Conv2d(512, class_num, 1)

        self.avg = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.MaxPool2d(2, 2)
            
    def forward(self, x):
        x = self.stem(x)
        
        f2 = self.Fires[0](x) + self.bp_conv1(x)
        f3 = self.Fires[1](f2) + f2
        
        f4 = self.Fires[2](f3) + self.bp_conv2(f3)
        f4 = self.maxpool(f4)

        f5 = self.Fires[3](f4) + f4
        f6 = self.Fires[4](f5) + self.bp_conv3(f5)
        f7 = self.Fires[5](f6) + f6
        f8 = self.Fires[6](f7) + self.bp_conv4(f7)
        f8 = self.maxpool(f8)

        f9 = self.Fires[7](f8) + f8
        c10 = self.conv10(f9)

        x = self.avg(c10)
        x = x.view(x.size(0), -1)

        return x
