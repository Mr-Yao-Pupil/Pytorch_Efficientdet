import torch.nn as nn
import torch
from model_file.layers import Swish, MemoryEfficientSwish, MaxPool2dStaticSamePadding, Conv2dStaticSamePadding
from model_file.efficientnet import EfficientNet as Main_Model
from utils.anchors import Anchors


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
            p3_out, p4_out, p5_out, p6_out, p7_out = self._forward_fast_attention(inputs)
        else:
            p3_out, p4_out, p5_out, p6_out, p7_out = self._forward(inputs)
        return p3_out, p4_out, p5_out, p6_out, p7_out

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
        if self.first_time:
            p3, p4, p5 = inputs
            p3_in = self.down_channel_p3(p3)

            p4_in_1 = self.down_channel_p4(p4)
            p4_in_2 = self.p4_down_channel_2(p4)

            p5_in_1 = self.down_channel_p5(p5)
            p5_in_2 = self.p5_down_channel_2(p5)

            # 原EfficientNet只有P5，通过额外增加下采样层生成p6和p7
            # 增加p6层输出
            p6_in = self._p5_2_p6(p5)
            # 增加p7层输出
            p7_in = self._p6_2_p7(p6_in)

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


class Box_Block(nn.Module):
    """返回物体框的网络模块"""

    def __init__(self, in_channels, num_anchors, num_layers, onnx_export=False):
        super(Box_Block, self).__init__()
        self.num_layers = num_layers
        self.conv_list = nn.ModuleList(
            [SeparableConvBlock(in_channels, in_channels, norm=False, activation=False) for i in range(num_layers)])
        self.bn_list = nn.ModuleList(
            [nn.ModuleList([nn.BatchNorm2d(in_channels, momentum=0.1, eps=1e-3)]) for i in range(5)])
        self.header = SeparableConvBlock(in_channels, num_anchors * 4, norm=False, activation=False)
        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, inputs):
        feats = []
        for feat, bn_list in zip(inputs, self.bn_list):
            for i, bn, conv in zip(range(self.num_layers), bn_list, self.conv_list):
                feat = self.swish(bn(conv(feat)))
            feat = self.header(feat)
            # # 轴排序
            # feat = feat.permute(0, 2, 3, 1)
            # # 使用contiguous返回可以使得Tensor view的数据类型
            # feat = feat.contiguous().view(feat.shape[0], -1, 4)
            feat = feat.reshape(feat.shape[0], 4, -1)
            feats.append(feat)
        feats = torch.cat(feats, dim=2)
        return feats


class Class_Block(nn.Module):
    """返回物体分类的模块"""

    def __init__(self, in_channels, num_anchors, num_classes, num_layers, onnx_export=False):
        super(Class_Block, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.conv_list = nn.ModuleList(
            [SeparableConvBlock(in_channels, in_channels, norm=False, activation=False) for i in range(num_layers)])
        self.bn_list = nn.ModuleList(
            [nn.ModuleList([nn.BatchNorm2d(in_channels, momentum=0.1, eps=1e-3) for i in range(num_layers)]) for i in
             range(5)])

        self.header = SeparableConvBlock(in_channels, num_anchors * num_classes, norm=False, activation=False)
        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, inputs):
        feats = []
        for feat, bn_list in zip(inputs, self.bn_list):
            for i, bn, conv in zip(range(self.num_layers), bn_list, self.conv_list):
                feat = self.swish(bn(conv(feat)))
            feat = self.header(feat)
            feat = feat.reshape(feat.shape[0], self.num_anchors, self.num_classes, feat.shape[2], feat.shape[3])
            feat = feat.reshape(feat.shape[0], self.num_classes, -1)
            feats.append(feat)
        feats = torch.cat(feats, dim=2)
        return feats.sigmoid()


class EfficientNet(nn.Module):
    def __init__(self, phi, load_weight=False):
        super(EfficientNet, self).__init__()
        """
        EfficientNet的网络主干部分

        :param phi: 网络选型b0~b7
        :param load_weights: 是否加载权重，默认False不加载
        """
        main_model = Main_Model.from_name(f'efficientnet-b{phi}', load_weight)
        del main_model._conv_head
        del main_model._bn1
        del main_model._avg_pooling
        del main_model._dropout
        del main_model._fc
        self.main_model = main_model

    def forward(self, inputs):
        x = self.main_model._swish(self.main_model._bn0(self.main_model._conv_stem(inputs)))

        feature_maps = []
        last_x = None
        for idx, block in enumerate(self.main_model._blocks):
            drop_connect_rate = self.main_model._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.main_model._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

            if block._depthwise.stride == [2, 2]:
                feature_maps.append(last_x)
            elif idx == len(self.main_model._blocks) - 1:
                feature_maps.append(x)
            last_x = x
        del last_x
        return feature_maps[1:]


class EfficientDet_BackBone(nn.Module):
    def __init__(self, num_classes=80, phi=0, load_weight=False):
        super(EfficientDet_BackBone, self).__init__()
        self.phi = phi
        self.backbone_phi = [0, 1, 2, 3, 4, 5, 6, 6]
        # self.backbone_phi = [0, 1, 2, 3, 4, 5, 6, 7]
        self.fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384]
        self.fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8]
        self.box_class_repeats = [3, 3, 3, 4, 4, 4, 5, 5]
        self.anchor_scale = [4., 4., 4., 4., 4., 4., 4., 5.]

        num_anchors = 9
        # 在著网络输出时，p3-p5的通道数目
        conv_channel_coef = {0: [40, 112, 320],
                             1: [40, 112, 320],
                             2: [48, 120, 352],
                             3: [48, 136, 384],
                             4: [56, 160, 448],
                             5: [64, 176, 512],
                             6: [72, 200, 576],
                             7: [72, 200, 576], }

        self.bifpn = nn.Sequential(
            *[BiFPN(self.fpn_num_filters[self.phi], conv_channel_coef[phi],
                    True if _ == 0 else False,
                    attention=True if phi < 6 else False) for _ in range(self.fpn_cell_repeats[phi])])

        self.num_classes = num_classes

        self.regressor = Box_Block(in_channels=self.fpn_num_filters[self.phi], num_anchors=num_anchors,
                                   num_layers=self.box_class_repeats[self.phi])
        self.classifier = Class_Block(in_channels=self.fpn_num_filters[self.phi], num_anchors=num_anchors,
                                      num_layers=self.box_class_repeats[self.phi], num_classes=num_classes)
        self.anchors = Anchors(anchor_scale=self.anchor_scale[phi])
        self.bockbone_net = EfficientNet(self.backbone_phi[phi], load_weight=load_weight)

    def freeze_bn(self):
        """固定BN层的可训练参数"""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, inputs):
        _, p3, p4, p5 = self.bockbone_net(inputs)

        features = self.bifpn((p3, p4, p5))

        # 检测部分
        regression = self.regressor(features)
        classification = self.classifier(features)
        anchors = self.anchors(inputs)

        return features, regression, classification, anchors
if __name__ == '__main__':
    model = EfficientDet_BackBone()
    test_inputs = torch.Tensor(1, 3, 256, 256)
    out = model(test_inputs)
    for i in out:
        try:
            print(i.shape)
        except:
            print(i[0].shape, i[1].shape)
