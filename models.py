import math
import torch
from torch import nn
from torch.nn.functional import relu, sigmoid

class StegaStampDecoder(nn.Module):
    def __init__(self, channels=1, image_in=False, deflat_out=False, shrink_rate=1):
        super(StegaStampDecoder, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        num_of_layers = 17
        if image_in:
            layers.append(nn.Conv2d(in_channels=6*channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        else:
            layers.append(nn.Conv2d(in_channels=2*channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        
        if deflat_out:
            while shrink_rate != 1:
                shrink_rate = shrink_rate // 2
                if shrink_rate == 1:
                    layers.append(nn.Conv2d(in_channels=features, out_channels=3, kernel_size=3, stride=2, padding=1, bias=False))
                else:
                    layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, stride=2, padding=1, bias=False))
        else:
            layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))

        self.dncnn = nn.Sequential(*layers)
    def forward(self, x):
        out = self.dncnn(x)
        return out