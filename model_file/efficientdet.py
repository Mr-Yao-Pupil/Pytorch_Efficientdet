import torch.nn as nn
import torch
from model_file.layers import Swish, MemoryEfficientSwish


class SeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, norm=True, activation=False, onne_export=False):
        super(SeparableConvBlock, self).__init__()
        if out_channels == None:
            out_channels = in_channels

        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1,
                                        groups=in_channels, bias=False)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.norm = norm
        if self.norm:
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)
        self.activation = activation
        if self.activation:
            self.swish = MemoryEfficientSwish if not onne_export else Swish()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        if self.norm:
            x = self.bn(x)
        if self.activation:
            x = self.swish(x)
        return x
