import torch
import torch.nn as nn
from torch.nn import functional as F
from model_file.layers import (round_filters, get_same_padding_conv2d)


class Main_Conv_Block(nn.Module):
    def __init__(self, bolck_args, global_params):
        super(Main_Conv_Block, self).__init__()
        self._block_args = bolck_args
        # 获得网络标准化参数
        self._bn_mom = 1 - global_params.batchnorm_momentum
        self._bn_eps = global_params.batchnorm_eps

        # 注意力机制的缩放比例
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        # 是否接入残差结构
        self.id_skip = bolck_args.id_skip

        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)
        # 卷积运算+标准化
        self._conv_stem = nn.Conv2d(input_channels, out_channels)



    def forward(self, x):
        pass
