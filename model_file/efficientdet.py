import torch.nn as nn
import torch
from model_file.layers import Swish, MemoryEfficientSwish, MaxPool2dStaticSamePadding, Conv2dStaticSamePadding


class SeparableConvBlock(nn.Module):
    """深度可分离卷积网络模块"""

    def __init__(self, in_channels, out_channels=None, norm=True, activation=False, onnx_export=False):
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
            self.swish = MemoryEfficientSwish if not onnx_export else Swish()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        if self.norm:
            x = self.bn(x)
        if self.activation:
            x = self.swish(x)
        return x


class BiFPN(nn.Module):
    def __init__(self, num_channels, conv_channels, first_time=False, epsilon=1e-4, onnx_export=False, attention=True):
        super(BiFPN, self).__init__()
        self.epsilon = epsilon
        # 分别对主网络的3， 4， 5， 6层实例化可分离卷积模块
        self.up_conv6 = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.up_conv5 = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.up_conv4 = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.up_conv3 = SeparableConvBlock(num_channels, onnx_export=onnx_export)

        # 分别对主网络的3， 4， 5， 6层实例化可分离卷积模块
        self.down_conv4 = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.down_conv5 = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.down_conv6 = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.down_conv7 = SeparableConvBlock(num_channels, onnx_export=onnx_export)

        # 分别对主网络3， 4， 5， 6模块输出实例化上采样
        self.upsample_p6 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample_p5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample_p4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample_p3 = nn.Upsample(scale_factor=2, mode='nearest')

        # 分别对主网络3， 4， 5， 6模块输出实例化下采样
        self.downsample_p4 = MaxPool2dStaticSamePadding(3, 2)
        self.downsample_p5 = MaxPool2dStaticSamePadding(3, 2)
        self.downsample_p6 = MaxPool2dStaticSamePadding(3, 2)
        self.downsample_p7 = MaxPool2dStaticSamePadding(3, 2)

        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()
        self.first_time = first_time

        if self.first_time:
            # 获取efficientNet的最后三层，对其通道进行下压缩
            self.down_channel_p5 = nn.Sequential(Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                                                 nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3))
            self.down_channel_p4 = nn.Sequential(Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
                                                 nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3))
            self.down_channel_p3 = nn.Sequential(Conv2dStaticSamePadding(conv_channels[0], num_channels, 1),
                                                 nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3))

            # 对输入的p5进行下采样
            self._p5_2_p6 = nn.Sequential(Conv2dStaticSamePadding(conv_channels[2], num_channels, 1, ),
                                          nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
                                          MaxPool2dStaticSamePadding(3, 2))
            self._p6_2_p7 = nn.Sequential(MaxPool2dStaticSamePadding(3, 2))

            # BiFPN第一轮时跳线那并不是
            self.p4_down_channel_2 = nn.Sequential(Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
                                                   nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3))
            self.p5_down_channel_2 = nn.Sequential(Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                                                   nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3))

        # 融合时使用的注意力
        self.attention = attention
        self.p6_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p6_w1_activate = nn.ReLU()
        self.p5_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p5_w1_activate = nn.ReLU()
        self.p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w1_activate = nn.ReLU()
        self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w1_activate = nn.ReLU()

        self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p4_w2_activate = nn.ReLU()
        self.p5_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p5_w2_activate = nn.ReLU()
        self.p6_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p6_w2_activate = nn.ReLU()
        self.p7_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p7_w2_activate = nn.ReLU()

    def forward(self, inputs):
        if self.attention:
            # out_p3, out_p4, out_p5, out_p6, out_p7 = \
            self._forward_fast_attention(inputs)

    def _forward_fast_attention(self, inputs):
        """ bifpn模块结构示意图
            p7_in -------------------------> P7_out -------->
               |--------------|                ↑
                              ↓                |
            p6_in ---------> P6_td --------> P6_out -------->
               |--------------|---------------↑ ↑
                              ↓                |
            p5_in -p5_in_1-> P5_td --------> P5_out -------->
               |---p5_in_2----|--------------↑ ↑
                              ↓                |
            P4_in -p4_in_1-> P4_td --------> p4_out -------->
               |---p4_in_2----|--------------↑ ↑
                             |---------------↓ |
            p5_in -------------------------> P3_out -------->
        """
        p3, p4, p5 = inputs
        if self.first_time:
            p3_in = self.down_channel_p3(p3)

            p4_in_1 = self.down_channel_p4(p4)
            p4_in_2 = self.p4_down_channel_2(p4)

            p5_in_1 = self.down_channel_p4(p5)
            p5_in_2 = self.p4_down_channel_2(p5)

            # 原EfficientNet只有P5，通过额外增加下采样层生成p6和p7
            # 增加p6层输出
            p6_in = self._p5_2_p6(p5)
            # 增加p7层输出
            p7_in = self._p5_2_p6(p6_in)

            # 用与p6_in和p7_in的注意力模块
            # 生成简单的通道注意力权重
            p6_w1 = self.p6_w1_activate(self.p5_w1)
            weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
            # 生成p6_1时网络关注的特征位置
            p6_td = self.up_conv6(self.swish(weight[0] * p6_in + weight[1] * self.upsample_p6(p7_in)))

            p5_w1 = self.p5_w1_activate(self.p5_w1)
            weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
            p5_td = self.up_conv5(self.swish(weight[0] * p5_in_1 + weight[1] * self.upsample_p5(p6_td)))

            p4_w1 = self.p4_w1_activate(self.p4_w1)
            weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
            p4_td = self.up_conv4(self.swish(weight[0] * p4_in_1 + weight[1] * self.upsample_p4(p5_td)))

            p3_w1 = self.p3_w1_activate(self.p3_w1)
            weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
            p3_out = self.up_conv3(self.swish(weight[0] * p3_in + weight[1] * self.upsample_p3(p4_td)))

            p4_w2 = self.p4_w2_activate(self.p4_w2)
            weight = p4_w2 / (torch.sum(p4_w1, dim=0) + self.epsilon)
            p4_out = self.down_conv4(
                self.swish(weight[0] * p4_in_2 + weight[1] * p4_td + weight[2] * self.downsample_p4(p3_out)))

            p5_w2 = self.p5_w2_activate(self.p5_w2)
            weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
            p5_out = self.down_conv5(
                self.swish(weight[0] * p5_in_2 + weight[1] * p5_td + weight[2] * self.downsample_p5(p4_out)))

            p6_w2 = self.p6_w2_activate(self.p6_w2)
            weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
            p6_out = self.down_conv6(
                self.swish(weight[0] * p6_in + weight[1] * p6_td + weight[2] * self.downsample_p6(p5_out)))

            p7_w2 = self.p7_w2_activate(self.p7_w2)
            weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
            p7_out = self.down_conv7(self.swish(weight[0] * p7_in + weight[1] * self.downsample_p7(p6_out)))

        else:
            p3_in, p4_in, p5_in, p6_in, p7_in = inputs

            p6_w1 = self.p6_w1_activate(self.p6_w1)
            weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
            p6_td = self.up_conv6(self.swish(weight[0] * p6_in + weight[1] * self.upsample_p6(p7_in)))

            p5_w1 = self.p5_w1_activate(self.p5_w1)
            weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
            p5_td = self.up_conv5(self.swish(weight[0] * p5_in + weight[1] * self.upsample_p5(p6_td)))

            p4_w1 = self.p4_w1_activate(self.p4_w1)
            weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
            p4_td = self.up_conv4(self.swish(weight[0] * p4_in + weight[1] * self.upsample_p4(p5_td)))

            p3_w1 = self.p3_w1_activate(self.p3_w1)
            weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
            p3_out = self.up_conv3(self.swish(weight[0] * p3_in + weight[1] * self.upsample_p3(p4_td)))

            p4_w2 = self.p4_w2_activate(self.p4_w2)
            weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
            p4_out = self.down_conv4(
                self.swish(weight[0] * p4_in + weight[1] * p4_td + weight[2] * self.downsample_p4(p3_out)))

            p5_w2 = self.p5_w2_activate(self.p5_w2)
            weight = p5_w2 / (torch.sum(p5_w2))
            p5_out = self.down_conv5(
                self.swish(weight[0] * p5_in + weight[1] * p5_td + weight[2] * self.downsample_p6(p4_out)))

            p6_w2 = self.p6_w2_activate(self.p6_w2)
            weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
            p6_out = self.down_conv6(
                self.swish(weight[0] * p6_in + weight[1] * p6_td + weight[2] * self.downsample_p6(p5_out)))

            p7_w2 = self.p7_w2_activate(self.p7_w2)
            weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
            p7_out = self.down_conv7(self.swish(weight[0] * p7_in + weight[1] * self.downsample_p7(p6_out)))

        return p3_out, p4_out, p5_out, p6_out, p7_out

    def _forward(self, inputs):
        """不加注意力的前向计算"""
        if self.first_time:
            p3, p4, p5 = inputs
            p3_in = self.down_channel_p3(p3)

            p4_in_1 = self.down_channel_p4(p4)
            p4_in_2 = self.p4_down_channel_2(p4)

            p5_in_1 = self.down_channel_p5(p5)
            p5_in_2 = self.p5_down_channel_2(p5)

            p6_in = self._p5_2_p6(p5)
            p7_in = self._p6_2_p7(p6_in)

            p6_td = self.up_conv6(self.swish(p6_in + self.upsample_p6(p7_in)))
            p5_td = self.up_conv5(self.swish(p5_in_1 + self.upsample_p5(p6_td)))
            p4_td = self.up_conv4(self.swish(p4_in_1 + self.upsample_p4(p5_td)))
            p3_out = self.up_conv3(self.swish(p3_in + self.upsample_p3(p4_td)))

            p4_out = self.down_conv4(self.swish(p4_in_2 + p4_td + self.downsample_p4(p3_out)))
            p5_out = self.down_conv5(self.swish(p5_in_2 + p5_td + self.downsample_p5(p4_out)))
            p6_out = self.down_conv6(self.swish(p6_in + p6_td + self.downsample_p6(p5_out)))
            p7_out = self.down_conv7(self.swish(p7_in + self.downsample_p7(p6_out)))

        else:
            p3_in, p4_in, p5_in, p6_in, p7_in = inputs

            p6_td = self.up_conv6(self.swish(p6_in + self.upsample_p6(p7_in)))
            p5_td = self.up_conv5(self.swish(p5_in + self.upsample_p5(p6_td)))
            p4_td = self.up_conv4(self.swish(p4_in + self.upsample_p4(p5_td)))
            p3_out = self.up_conv3(self.swish(p3_in + self.upsample_p3(p4_td)))

            p4_out = self.down_conv4(self.swish(p4_in + p4_td + self.downsample_p4(p3_out)))
            p5_out = self.down_conv5(self.swish(p5_in + p5_td + self.downsample_p5(p4_out)))
            p6_out = self.down_conv6(self.swish(p6_in + p6_td + self.downsample_p6(p5_out)))
            p7_out = self.down_conv6(self.swish(p7_in + self.downsample_p7(p6_out)))
        return p3_out, p4_out, p5_out, p6_out, p7_out
